"""OKX Demo Trading adapter (via ccxt).

OKX "Demo Trading" is a paper environment on the production API host,
switched on by the x-simulated-trading header (ccxt sandbox mode). Reachable
from US datacenter IPs (GitHub Actions), unlike Bybit, and unlike the Binance
futures testnet it does not require a Binance account to register.

OKX derivatives quirks handled here:
- Orders are placed in CONTRACTS, not coins. Each swap has a contract value
  (ctVal, in base coin). risk.calc_order still works in coins: instrument()
  reports qty_step = lotSz*ctVal and min_qty = minSz*ctVal (both in coins),
  and _to_contracts() converts the resulting coin qty for order placement.
- SL/TP are attached to the entry order (takeProfitPrice/stopLossPrice params,
  position-level, market execution) — they die together with the position,
  like on Bybit. cancel_stray_orders() is a best-effort safety net.
"""

import logging
import time
from decimal import ROUND_DOWN, Decimal

import pandas as pd

from market_data import MarketData  # reuse indicator implementations

logger = logging.getLogger("bot.okx")


def create_session(cfg):
    import ccxt

    ex = ccxt.okx({
        "apiKey": cfg.okx_api_key,
        "secret": cfg.okx_api_secret,
        "password": cfg.okx_passphrase,
        "enableRateLimit": True,
    })
    ex.set_sandbox_mode(True)  # x-simulated-trading: 1 -> Demo Trading
    return ex


def _unified(symbol: str) -> str:
    """'BTCUSDT' -> ccxt unified swap symbol 'BTC/USDT:USDT'."""
    base = symbol[:-4] if symbol.upper().endswith("USDT") else symbol
    return f"{base}/USDT:USDT"


class OkxMarketData:
    """Same interface as market_data.MarketData."""

    def __init__(self, ex):
        self.ex = ex

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        try:
            if not self.ex.markets:
                self.ex.load_markets()
            tf = {"15": "15m", "60": "1h"}.get(str(interval), f"{interval}m")
            rows = self.ex.fetch_ohlcv(
                _unified(symbol), timeframe=tf, limit=min(int(limit), 300)
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


class OkxTrader:
    """Same interface as trader.Trader (+ cancel_stray_orders)."""

    def __init__(self, ex, cfg):
        self.ex = ex
        self.cfg = cfg
        self._instruments: dict[str, dict] = {}  # + raw ctVal/lotSz for conversion

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        self.ex.load_markets()
        for symbol in self.cfg.symbols:
            try:
                self.ex.set_leverage(
                    self.cfg.leverage, _unified(symbol), params={"mgnMode": "cross"}
                )
                logger.info("leverage set to %sx for %s", self.cfg.leverage, symbol)
            except Exception as exc:
                logger.warning("set_leverage failed for %s: %s", symbol, exc)
            self._cache_instrument(symbol)

    def _cache_instrument(self, symbol: str) -> None:
        try:
            market = self.ex.market(_unified(symbol))
            info = market.get("info", {})
            ct_val = Decimal(str(info.get("ctVal", "0")))
            lot_sz = Decimal(str(info.get("lotSz", "0")))
            min_sz = Decimal(str(info.get("minSz", "0")))
            tick_sz = float(info.get("tickSz", 0) or 0)
            if ct_val <= 0 or lot_sz <= 0 or tick_sz <= 0:
                logger.error("bad instrument info for %s: %s", symbol, info)
                return
            self._instruments[symbol] = {
                # coin-denominated filters for risk.calc_order
                "qty_step": float(lot_sz * ct_val),
                "min_qty": float((min_sz or lot_sz) * ct_val),
                "tick_size": tick_sz,
                # raw values for coin -> contracts conversion
                "_ct_val": ct_val,
                "_lot_sz": lot_sz,
            }
            logger.info("instrument cached for %s: %s",
                        symbol, self._instruments[symbol])
        except Exception as exc:
            logger.error("instrument info failed for %s: %s", symbol, exc)

    def instrument(self, symbol: str) -> dict | None:
        if symbol not in self._instruments:
            self._cache_instrument(symbol)
        return self._instruments.get(symbol)

    def _to_contracts(self, symbol: str, qty_coins: float) -> float:
        inst = self.instrument(symbol)
        if inst is None:
            raise RuntimeError(f"no instrument info for {symbol}")
        ct_val: Decimal = inst["_ct_val"]
        lot_sz: Decimal = inst["_lot_sz"]
        contracts = Decimal(str(qty_coins)) / ct_val
        steps = (contracts / lot_sz).to_integral_value(rounding=ROUND_DOWN)
        return float(steps * lot_sz)

    # -------------------------------------------------------------- positions

    def get_positions(self) -> dict:
        """Raises on API failure (callers skip the pass), like Trader."""
        positions = self.ex.fetch_positions(params={"instType": "SWAP"})
        out: dict[str, dict] = {}
        for pos in positions:
            contracts = float(pos.get("contracts") or 0)
            if contracts <= 0:
                continue
            inst_id = (pos.get("info") or {}).get("instId", "")  # BTC-USDT-SWAP
            market_id = inst_id.replace("-SWAP", "").replace("-", "")
            ct_val = float(
                (self.ex.market(pos["symbol"]).get("info") or {}).get("ctVal", 0) or 0
            )
            out[market_id] = {
                "side": "Buy" if pos.get("side") == "long" else "Sell",
                "qty": contracts * ct_val if ct_val else contracts,
                "entry": float(pos.get("entryPrice") or 0),
                "unrealized": float(pos.get("unrealizedPnl") or 0),
            }
        return out

    def open_position(self, symbol, side, qty, sl, tp) -> dict:
        """Market entry with attached position-level SL/TP (market execution)."""
        uni = _unified(symbol)
        contracts = self._to_contracts(symbol, qty)
        if contracts <= 0:
            raise RuntimeError(f"qty {qty} converts to 0 contracts for {symbol}")
        ccxt_side = "buy" if side == "Buy" else "sell"
        try:
            entry = self.ex.create_order(
                uni, "market", ccxt_side, contracts, None,
                {
                    "tdMode": "cross",
                    "stopLossPrice": sl,
                    "takeProfitPrice": tp,
                },
            )
        except Exception:
            logger.exception("entry with attached SL/TP failed for %s", symbol)
            raise
        logger.info("opened %s %s qty=%s (contracts=%s) sl=%s tp=%s",
                    side, symbol, qty, contracts, sl, tp)
        return entry

    def close_position(self, symbol: str) -> bool:
        pos = self.get_positions().get(symbol)
        if not pos:
            logger.warning("close_position: no open position for %s", symbol)
            return False
        uni = _unified(symbol)
        contracts = self._to_contracts(symbol, pos["qty"])
        opp = "sell" if pos["side"] == "Buy" else "buy"
        self.ex.create_order(
            uni, "market", opp, contracts, None,
            {"tdMode": "cross", "reduceOnly": True},
        )
        self.cancel_stray_orders(symbol)
        logger.info("closed position %s (%s qty=%s)", symbol, pos["side"], pos["qty"])
        return True

    def cancel_stray_orders(self, symbol: str) -> None:
        """Best-effort cleanup of pending algo (TP/SL) orders for the symbol.

        Attached SL/TP on OKX are position-level and normally disappear with
        the position; this catches leftovers after partial fills or API races.
        """
        inst_id = f"{symbol[:-4]}-USDT-SWAP" if symbol.upper().endswith("USDT") else symbol
        try:
            for ord_type in ("oco", "conditional"):
                resp = self.ex.privateGetTradeOrdersAlgoPending(
                    {"instType": "SWAP", "instId": inst_id, "ordType": ord_type}
                )
                algos = resp.get("data", []) if isinstance(resp, dict) else []
                if algos:
                    self.ex.privatePostTradeCancelAlgos([
                        {"algoId": a["algoId"], "instId": inst_id} for a in algos
                    ])
        except Exception:
            logger.debug("cancel_stray_orders noop/failed for %s", symbol)

    # ---------------------------------------------------------------- account

    def get_equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            data = (bal.get("info", {}).get("data") or [{}])[0]
            if data.get("totalEq") is not None:
                return float(data["totalEq"])
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
        """Realized PnL of the most recent closed position for the symbol."""
        inst_id = f"{symbol[:-4]}-USDT-SWAP" if symbol.upper().endswith("USDT") else symbol
        for attempt in range(retries):
            try:
                resp = self.ex.privateGetAccountPositionsHistory(
                    {"instType": "SWAP", "instId": inst_id, "limit": "10"}
                )
                for rec in resp.get("data", []):  # newest first
                    u_time = float(rec.get("uTime") or 0)
                    if since_ms is not None and u_time < since_ms:
                        continue
                    pnl = rec.get("realizedPnl", rec.get("pnl"))
                    if pnl is not None:
                        return float(pnl)
            except Exception as exc:
                logger.error("positions-history failed for %s: %s", symbol, exc)
            if attempt < retries - 1:
                time.sleep(delay_s)
        logger.warning("no realized-pnl record for %s (since_ms=%s)", symbol, since_ms)
        return None
