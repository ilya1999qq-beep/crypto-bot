"""Order execution and account state for Bybit demo trading (USDT linear perps).

Thin synchronous wrapper around pybit.unified_trading.HTTP. All methods are
blocking — call them via asyncio.to_thread(...) from async code.
"""

import logging
import time

logger = logging.getLogger("bot.trader")

LEVERAGE_NOT_MODIFIED = 110043  # Bybit ErrCode: leverage already set to this value


def _f(value, default: float = 0.0) -> float:
    """Parse a Bybit string number defensively."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Trader:
    def __init__(self, session, cfg):
        self.session = session
        self.cfg = cfg
        self._instruments: dict[str, dict] = {}  # symbol -> {"qty_step","min_qty","tick_size"}

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        """Set leverage and cache instrument filters for every configured symbol."""
        for symbol in self.cfg.symbols:
            self._set_leverage(symbol)
            self._cache_instrument(symbol)

    def _set_leverage(self, symbol: str) -> None:
        lev = str(self.cfg.leverage)
        try:
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=lev,
                sellLeverage=lev,
            )
            logger.info("leverage set to %sx for %s", lev, symbol)
        except Exception as exc:  # pybit raises InvalidRequestError with the code in message
            code = getattr(exc, "status_code", None)
            if code == LEVERAGE_NOT_MODIFIED or str(LEVERAGE_NOT_MODIFIED) in str(exc):
                logger.debug("leverage already %sx for %s (110043)", lev, symbol)
            else:
                logger.error("set_leverage failed for %s: %s", symbol, exc)

    def _cache_instrument(self, symbol: str) -> None:
        try:
            resp = self.session.get_instruments_info(category="linear", symbol=symbol)
            items = resp.get("result", {}).get("list", [])
            if not items:
                logger.error("no instrument info returned for %s", symbol)
                return
            info = items[0]
            lot = info.get("lotSizeFilter", {})
            price = info.get("priceFilter", {})
            self._instruments[symbol] = {
                "qty_step": _f(lot.get("qtyStep"), 0.001),
                "min_qty": _f(lot.get("minOrderQty"), 0.001),
                "tick_size": _f(price.get("tickSize"), 0.01),
            }
            logger.info("instrument cached for %s: %s", symbol, self._instruments[symbol])
        except Exception as exc:
            logger.error("get_instruments_info failed for %s: %s", symbol, exc)

    def instrument(self, symbol: str) -> dict | None:
        """Return cached filters; fetch lazily if setup missed this symbol.

        Returns None when real exchange filters are unavailable — generic
        defaults would mis-round qty/SL/TP for most symbols and every order
        would be rejected, so callers must skip the entry instead.
        """
        if symbol not in self._instruments:
            self._cache_instrument(symbol)
        return self._instruments.get(symbol)

    # -------------------------------------------------------------- positions

    def get_positions(self) -> dict:
        """Open linear USDT positions: {symbol: {side, qty, entry, unrealized}}.

        Raises on any API/network failure instead of returning {} so callers
        can distinguish "no open positions" from "snapshot unavailable".
        Never reconcile or open entries off a failed snapshot.
        """
        try:
            resp = self.session.get_positions(category="linear", settleCoin="USDT")
        except Exception as exc:
            logger.error("get_positions failed: %s", exc)
            raise
        out: dict[str, dict] = {}
        for pos in resp.get("result", {}).get("list", []):
            size = _f(pos.get("size"))
            if size <= 0:
                continue
            out[pos.get("symbol", "")] = {
                "side": pos.get("side", ""),
                "qty": size,
                "entry": _f(pos.get("avgPrice")),
                "unrealized": _f(pos.get("unrealisedPnl")),
            }
        return out

    def open_position(self, symbol, side, qty, sl, tp) -> dict:
        """Market entry with attached SL/TP. side: 'Buy' | 'Sell'."""
        resp = self.session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            takeProfit=str(tp),
            stopLoss=str(sl),
            positionIdx=0,
        )
        logger.info("opened %s %s qty=%s sl=%s tp=%s", side, symbol, qty, sl, tp)
        return resp

    def close_position(self, symbol: str) -> bool:
        """Close the full position with a reduce-only market order.

        Returns True if a close order was actually placed, False if there is
        no open position for the symbol. Raises if the position lookup or the
        order placement fails, so callers never mistake an error for success.
        """
        pos = self.get_positions().get(symbol)
        if not pos:
            logger.warning("close_position: no open position for %s", symbol)
            return False
        opposite = "Sell" if pos["side"] == "Buy" else "Buy"
        self.session.place_order(
            category="linear",
            symbol=symbol,
            side=opposite,
            orderType="Market",
            qty=str(pos["qty"]),
            reduceOnly=True,
            positionIdx=0,
        )
        logger.info("closed position %s (%s qty=%s)", symbol, pos["side"], pos["qty"])
        return True

    def cancel_stray_orders(self, symbol: str) -> None:
        """No-op on Bybit: attached SL/TP die together with the position."""

    # ---------------------------------------------------------------- account

    def get_equity(self) -> float:
        try:
            resp = self.session.get_wallet_balance(accountType="UNIFIED")
            items = resp.get("result", {}).get("list", [])
            if not items:
                return 0.0
            return _f(items[0].get("totalEquity"))
        except Exception as exc:
            logger.error("get_wallet_balance failed: %s", exc)
            return 0.0

    def last_closed_pnl(
        self,
        symbol: str,
        since_ms: int | None = None,
        retries: int = 3,
        delay_s: float = 1.5,
    ) -> float | None:
        """Realized PnL of the most recent closed trade for the symbol.

        Bybit writes the closed-pnl record asynchronously after the fill, so a
        naive limit=1 query can return the PREVIOUS trade's record. When
        since_ms is given, only a record with updatedTime >= since_ms is
        accepted (since_ms is also passed as startTime to filter server-side),
        retrying briefly until it appears. Returns None if no matching record
        shows up. Blocking (sleeps between retries) — call via asyncio.to_thread.
        """
        for attempt in range(retries):
            try:
                kwargs: dict = {"category": "linear", "symbol": symbol, "limit": 1}
                if since_ms is not None:
                    kwargs["startTime"] = since_ms
                resp = self.session.get_closed_pnl(**kwargs)
                items = resp.get("result", {}).get("list", [])
                if items:
                    item = items[0]
                    if since_ms is None or _f(item.get("updatedTime")) >= since_ms:
                        return _f(item.get("closedPnl"))
            except Exception as exc:
                logger.error("get_closed_pnl failed for %s: %s", symbol, exc)
            if attempt < retries - 1:
                time.sleep(delay_s)
        logger.warning(
            "no matching closed-pnl record for %s (since_ms=%s)", symbol, since_ms
        )
        return None
