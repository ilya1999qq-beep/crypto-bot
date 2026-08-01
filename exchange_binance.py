"""Binance USDT-M Futures Testnet adapter (via ccxt).

Mirrors the public interfaces of market_data.MarketData and trader.Trader so
strategy/risk/journal code stays exchange-agnostic. Chosen because the Binance
futures testnet is reachable from US datacenter IPs (GitHub Actions runners),
unlike every Bybit domain.

Key semantic difference vs Bybit: Binance futures has no "attached" SL/TP on
the entry order — they are separate conditional orders (STOP_MARKET /
TAKE_PROFIT_MARKET with closePosition=true). When one of them fires, the other
stays behind, so callers must invoke cancel_stray_orders() for symbols whose
position is gone (the reconcile step does this).
"""

import logging
import time

import pandas as pd

from market_data import MarketData  # reuse the indicator implementations

logger = logging.getLogger("bot.binance")


def _interval_to_tf(interval: str) -> str:
    """Bybit-style interval ('15', '60') -> ccxt timeframe ('15m', '1h')."""
    mapping = {"1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
               "60": "1h", "120": "2h", "240": "4h", "D": "1d"}
    return mapping.get(str(interval), f"{interval}m")


def create_session(cfg):
    """Authenticated ccxt binanceusdm client in sandbox (testnet) mode."""
    import ccxt

    ex = ccxt.binanceusdm({
        "apiKey": cfg.binance_api_key,
        "secret": cfg.binance_api_secret,
        "enableRateLimit": True,
    })
    ex.set_sandbox_mode(True)
    return ex


class BinanceMarketData:
    """Same interface as market_data.MarketData."""

    def __init__(self, ex):
        self.ex = ex

    def _unified(self, symbol: str) -> str:
        m = self.ex.markets_by_id.get(symbol)
        if isinstance(m, list):
            m = m[0] if m else None
        return m["symbol"] if m else symbol

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        try:
            if not self.ex.markets:
                self.ex.load_markets()
            rows = self.ex.fetch_ohlcv(
                self._unified(symbol), timeframe=_interval_to_tf(interval),
                limit=limit,
            )
            if not rows or len(rows) < 2:
                return MarketData._empty_frame()
            rows = rows[:-1]  # drop the still-open candle
            df = pd.DataFrame(
                rows, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
            return df.astype(float)
        except Exception as exc:
            logger.error("fetch_ohlcv failed for %s: %s", symbol, exc)
            return MarketData._empty_frame()

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        return MarketData.add_indicators(df)

    def get_frames(self, symbol: str, cfg):
        df15 = self.add_indicators(self.get_klines(symbol, cfg.timeframe))
        df1h = self.add_indicators(self.get_klines(symbol, cfg.trend_timeframe))
        return df15, df1h


class BinanceTrader:
    """Same interface as trader.Trader (+ cancel_stray_orders)."""

    def __init__(self, ex, cfg):
        self.ex = ex
        self.cfg = cfg
        self._instruments: dict[str, dict] = {}

    def _unified(self, symbol: str) -> str:
        m = self.ex.markets_by_id.get(symbol)
        if isinstance(m, list):
            m = m[0] if m else None
        return m["symbol"] if m else symbol

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        self.ex.load_markets()
        for symbol in self.cfg.symbols:
            try:
                self.ex.set_leverage(self.cfg.leverage, self._unified(symbol))
                logger.info("leverage set to %sx for %s", self.cfg.leverage, symbol)
            except Exception as exc:
                logger.warning("set_leverage failed for %s: %s", symbol, exc)
            self._cache_instrument(symbol)

    def _cache_instrument(self, symbol: str) -> None:
        try:
            market = self.ex.markets_by_id.get(symbol)
            if isinstance(market, list):
                market = market[0] if market else None
            if not market:
                logger.error("no market info for %s", symbol)
                return
            filters = {
                f["filterType"]: f
                for f in market.get("info", {}).get("filters", [])
            }
            lot = filters.get("LOT_SIZE", {})
            price = filters.get("PRICE_FILTER", {})
            self._instruments[symbol] = {
                "qty_step": float(lot.get("stepSize", 0) or 0),
                "min_qty": float(lot.get("minQty", 0) or 0),
                "tick_size": float(price.get("tickSize", 0) or 0),
            }
            logger.info("instrument cached for %s: %s",
                        symbol, self._instruments[symbol])
        except Exception as exc:
            logger.error("instrument info failed for %s: %s", symbol, exc)

    def instrument(self, symbol: str) -> dict | None:
        if symbol not in self._instruments:
            self._cache_instrument(symbol)
        inst = self._instruments.get(symbol)
        if inst and inst["qty_step"] > 0 and inst["tick_size"] > 0:
            return inst
        return None

    # -------------------------------------------------------------- positions

    def get_positions(self) -> dict:
        """Raises on API failure (callers must skip the pass), like Trader."""
        positions = self.ex.fetch_positions()
        out: dict[str, dict] = {}
        for pos in positions:
            qty = float(pos.get("contracts") or 0)
            if qty <= 0:
                continue
            market_id = (pos.get("info") or {}).get("symbol") or pos.get("symbol", "")
            out[market_id] = {
                "side": "Buy" if pos.get("side") == "long" else "Sell",
                "qty": qty,
                "entry": float(pos.get("entryPrice") or 0),
                "unrealized": float(pos.get("unrealizedPnl") or 0),
            }
        return out

    def open_position(self, symbol, side, qty, sl, tp) -> dict:
        """Market entry + separate closePosition SL/TP conditional orders.

        If a protective order cannot be placed, the fresh position is closed
        immediately — never leave an unprotected position on the exchange.
        """
        uni = self._unified(symbol)
        ccxt_side = "buy" if side == "Buy" else "sell"
        opp = "sell" if side == "Buy" else "buy"

        entry = self.ex.create_order(uni, "market", ccxt_side, qty)
        try:
            self.ex.create_order(
                uni, "STOP_MARKET", opp, None, None,
                {"stopPrice": sl, "closePosition": True},
            )
            self.ex.create_order(
                uni, "TAKE_PROFIT_MARKET", opp, None, None,
                {"stopPrice": tp, "closePosition": True},
            )
        except Exception:
            logger.exception(
                "failed to place SL/TP for %s — emergency closing position", symbol
            )
            try:
                self.ex.cancel_all_orders(uni)
                self.ex.create_order(
                    uni, "market", opp, qty, None, {"reduceOnly": True}
                )
            except Exception:
                logger.exception("emergency close failed for %s", symbol)
            raise
        logger.info("opened %s %s qty=%s sl=%s tp=%s", side, symbol, qty, sl, tp)
        return entry

    def close_position(self, symbol: str) -> bool:
        pos = self.get_positions().get(symbol)
        if not pos:
            logger.warning("close_position: no open position for %s", symbol)
            return False
        uni = self._unified(symbol)
        opp = "sell" if pos["side"] == "Buy" else "buy"
        self.ex.create_order(
            uni, "market", opp, pos["qty"], None, {"reduceOnly": True}
        )
        try:
            self.ex.cancel_all_orders(uni)  # remove leftover SL/TP orders
        except Exception:
            logger.exception("cancel_all_orders failed for %s", symbol)
        logger.info("closed position %s (%s qty=%s)", symbol, pos["side"], pos["qty"])
        return True

    def cancel_stray_orders(self, symbol: str) -> None:
        """Cancel leftover SL/TP conditionals after the position is gone."""
        try:
            self.ex.cancel_all_orders(self._unified(symbol))
        except Exception:
            logger.exception("cancel_stray_orders failed for %s", symbol)

    # ---------------------------------------------------------------- account

    def get_equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            info = bal.get("info", {})
            for key in ("totalMarginBalance", "totalWalletBalance"):
                if info.get(key) is not None:
                    return float(info[key])
            return float(bal.get("USDT", {}).get("total") or 0.0)
        except Exception as exc:
            logger.error("fetch_balance failed: %s", exc)
            return 0.0

    def last_closed_pnl(
        self,
        symbol: str,
        since_ms: int | None = None,
        retries: int = 3,
        delay_s: float = 1.5,
    ) -> float | None:
        """Realized PnL from Binance income history; same contract as Trader."""
        for attempt in range(retries):
            try:
                params = {
                    "symbol": symbol,
                    "incomeType": "REALIZED_PNL",
                    "limit": 20,
                }
                if since_ms is not None:
                    params["startTime"] = since_ms
                records = self.ex.fapiPrivateGetIncome(params)
                if records:
                    total = sum(float(r.get("income") or 0) for r in records)
                    return total
            except Exception as exc:
                logger.error("income query failed for %s: %s", symbol, exc)
            if attempt < retries - 1:
                time.sleep(delay_s)
        logger.warning("no realized-pnl records for %s (since_ms=%s)",
                       symbol, since_ms)
        return None
