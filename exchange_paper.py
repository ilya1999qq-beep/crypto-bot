"""Paper-trading engine: real market prices, simulated execution.

Exists because every real demo/testnet venue turned out to be unusable for
this setup: all Bybit domains are geo-blocked from US GitHub-runner IPs, the
Binance futures testnet needs a Binance account, and OKX/Bitget demo accounts
require identity verification the user cannot pass. Public market data, on the
other hand, needs no account anywhere.

So the "exchange" here is local: prices and candles come from OKX's public
endpoints (real, live), while positions, SL/TP fills, fees and PnL are
simulated and persisted in the journal's sqlite state. The trader exposes the
same interface as trader.Trader, so strategy/risk/journal/runner code is
unchanged.

Fill model (deliberately pessimistic rather than flattering):
- entry at the current last price, with a taker fee,
- SL/TP resolved against real 1-minute candles that elapsed since the last
  check, in chronological order; if one candle's range covers both levels,
  the stop-loss is assumed to have been hit first,
- exit at exactly the SL/TP price, with a taker fee.
"""

import json
import logging
import time

from exchange_okx import OkxMarketData, _unified

logger = logging.getLogger("bot.paper")

POSITIONS_KEY = "paper_positions"
CLOSED_KEY = "paper_closed"
TAKER_FEE = 0.0005          # 0.05% per side, typical futures taker fee
CLOSED_HISTORY_LIMIT = 50


def create_session(cfg=None):
    """Public, unauthenticated OKX client — no API keys involved."""
    import ccxt

    return ccxt.okx({"enableRateLimit": True})


class PaperMarketData(OkxMarketData):
    """Real public candles; identical interface to market_data.MarketData."""


class PaperTrader:
    """Simulated exchange with the same interface as trader.Trader."""

    def __init__(self, ex, cfg, journal):
        self.ex = ex
        self.cfg = cfg
        self.journal = journal
        self._instruments: dict[str, dict] = {}

    # ------------------------------------------------------------- persistence

    def _load(self, key: str, default):
        raw = self.journal.get_state(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.error("corrupt state for %s, resetting", key)
            return default

    def _save(self, key: str, value) -> None:
        self.journal.set_state(key, json.dumps(value))

    def _positions_raw(self) -> dict:
        return self._load(POSITIONS_KEY, {})

    def _closed(self) -> list:
        return self._load(CLOSED_KEY, [])

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        self.ex.load_markets()
        for symbol in self.cfg.symbols:
            self._cache_instrument(symbol)

    def _cache_instrument(self, symbol: str) -> None:
        """Realistic lot/tick filters, taken from the real OKX contract specs."""
        try:
            info = self.ex.market(_unified(symbol)).get("info", {})
            ct_val = float(info.get("ctVal", 0) or 0)
            lot_sz = float(info.get("lotSz", 0) or 0)
            min_sz = float(info.get("minSz", 0) or 0)
            tick_sz = float(info.get("tickSz", 0) or 0)
            if ct_val <= 0 or lot_sz <= 0 or tick_sz <= 0:
                logger.error("bad instrument info for %s: %s", symbol, info)
                return
            self._instruments[symbol] = {
                "qty_step": lot_sz * ct_val,
                "min_qty": (min_sz or lot_sz) * ct_val,
                "tick_size": tick_sz,
            }
        except Exception as exc:
            logger.error("instrument info failed for %s: %s", symbol, exc)

    def instrument(self, symbol: str) -> dict | None:
        if symbol not in self._instruments:
            self._cache_instrument(symbol)
        return self._instruments.get(symbol)

    # -------------------------------------------------------------- settlement

    def _fetch_1m(self, symbol: str, since_ms: int) -> list:
        """Real 1-minute candles strictly after since_ms (oldest first)."""
        rows = self.ex.fetch_ohlcv(
            _unified(symbol), timeframe="1m", since=since_ms, limit=300
        )
        return [r for r in rows if r[0] > since_ms]

    def _settle(self, symbol: str, pos: dict) -> dict | None:
        """Replay elapsed 1m candles; return the close record if SL/TP hit."""
        since = int(pos.get("last_check_ms") or pos["opened_ms"])
        candles = self._fetch_1m(symbol, since)
        if not candles:
            return None

        sl, tp = float(pos["sl"]), float(pos["tp"])
        is_long = pos["side"] == "Buy"
        for ts, _o, high, low, _c, _v in candles:
            hit_sl = low <= sl if is_long else high >= sl
            hit_tp = high >= tp if is_long else low <= tp
            if hit_sl or hit_tp:
                # Pessimistic: when a single candle spans both levels, assume
                # the stop was reached first.
                exit_price = sl if hit_sl else tp
                return {
                    "symbol": symbol,
                    "exit": exit_price,
                    "ts_ms": int(ts),
                    "reason": "SL" if hit_sl else "TP",
                }
        pos["last_check_ms"] = int(candles[-1][0])
        return None

    def _realize(self, pos: dict, exit_price: float) -> float:
        qty, entry = float(pos["qty"]), float(pos["entry"])
        direction = 1.0 if pos["side"] == "Buy" else -1.0
        gross = (exit_price - entry) * qty * direction
        fees = (entry + exit_price) * qty * TAKER_FEE
        return gross - fees

    # --------------------------------------------------------------- positions

    def get_positions(self) -> dict:
        """Settle pending SL/TP first, then report what is still open.

        Raises on data-fetch failure so the caller skips the pass instead of
        acting on a snapshot that wrongly looks empty.
        """
        positions = self._positions_raw()
        if not positions:
            return {}

        closed = self._closed()
        still_open: dict[str, dict] = {}
        dirty = False

        for symbol, pos in list(positions.items()):
            fill = self._settle(symbol, pos)   # may raise -> pass is skipped
            if fill is None:
                still_open[symbol] = pos
                dirty = True                    # last_check_ms advanced
                continue
            pnl = self._realize(pos, fill["exit"])
            closed.append({
                "symbol": symbol,
                "pnl": pnl,
                "exit": fill["exit"],
                "ts_ms": fill["ts_ms"],
                "reason": fill["reason"],
            })
            logger.info("paper %s closed by %s at %s, pnl=%.2f",
                        symbol, fill["reason"], fill["exit"], pnl)
            dirty = True

        if dirty:
            self._save(POSITIONS_KEY, still_open)
            self._save(CLOSED_KEY, closed[-CLOSED_HISTORY_LIMIT:])

        prices = {}
        for symbol in still_open:
            try:
                prices[symbol] = float(
                    self.ex.fetch_ticker(_unified(symbol))["last"]
                )
            except Exception:
                logger.exception("ticker failed for %s", symbol)

        out = {}
        for symbol, pos in still_open.items():
            price = prices.get(symbol, float(pos["entry"]))
            direction = 1.0 if pos["side"] == "Buy" else -1.0
            out[symbol] = {
                "side": pos["side"],
                "qty": float(pos["qty"]),
                "entry": float(pos["entry"]),
                "unrealized": (price - float(pos["entry"]))
                * float(pos["qty"]) * direction,
            }
        return out

    def open_position(self, symbol, side, qty, sl, tp) -> dict:
        price = float(self.ex.fetch_ticker(_unified(symbol))["last"])
        now_ms = int(time.time() * 1000)
        positions = self._positions_raw()
        positions[symbol] = {
            "side": side,
            "qty": float(qty),
            "entry": price,
            "sl": float(sl),
            "tp": float(tp),
            "opened_ms": now_ms,
            "last_check_ms": now_ms,
        }
        self._save(POSITIONS_KEY, positions)
        logger.info("paper opened %s %s qty=%s at %s sl=%s tp=%s",
                    side, symbol, qty, price, sl, tp)
        return {"paper": True, "entry": price}

    def close_position(self, symbol: str) -> bool:
        positions = self._positions_raw()
        pos = positions.get(symbol)
        if not pos:
            logger.warning("close_position: no open paper position for %s", symbol)
            return False
        price = float(self.ex.fetch_ticker(_unified(symbol))["last"])
        pnl = self._realize(pos, price)
        del positions[symbol]
        self._save(POSITIONS_KEY, positions)
        closed = self._closed()
        closed.append({
            "symbol": symbol,
            "pnl": pnl,
            "exit": price,
            "ts_ms": int(time.time() * 1000),
            "reason": "manual",
        })
        self._save(CLOSED_KEY, closed[-CLOSED_HISTORY_LIMIT:])
        logger.info("paper closed %s at %s, pnl=%.2f", symbol, price, pnl)
        return True

    def cancel_stray_orders(self, symbol: str) -> None:
        """Nothing to cancel: simulated SL/TP live with the position."""

    # ----------------------------------------------------------------- account

    def get_equity(self) -> float:
        realized = sum(float(c.get("pnl") or 0.0) for c in self._closed())
        unrealized = 0.0
        try:
            for pos in self.get_positions().values():
                unrealized += float(pos.get("unrealized") or 0.0)
        except Exception:
            logger.exception("equity: position valuation failed")
        return self.cfg.budget_usdt + realized + unrealized

    def last_closed_pnl(
        self,
        symbol: str,
        since_ms: int | None = None,
        retries: int = 3,
        delay_s: float = 1.5,
    ) -> float | None:
        for record in reversed(self._closed()):
            if record.get("symbol") != symbol:
                continue
            if since_ms is not None and float(record.get("ts_ms") or 0) < since_ms:
                continue
            return float(record.get("pnl") or 0.0)
        logger.warning("no paper close record for %s (since_ms=%s)", symbol, since_ms)
        return None
