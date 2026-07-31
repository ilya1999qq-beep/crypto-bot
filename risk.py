"""Risk management: position limits, daily loss guard, order sizing."""

import logging
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation

logger = logging.getLogger("bot.risk")


def _floor_to_step(value: float, step: float) -> float:
    """Round value DOWN to the nearest multiple of step using Decimal math."""
    try:
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        if d_step <= 0:
            return float(d_value)
        floored = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        return float(floored)
    except (InvalidOperation, ValueError):
        return 0.0


def _round_to_tick(value: float, tick: float) -> float:
    """Round value to the nearest multiple of tick using Decimal math."""
    try:
        d_value = Decimal(str(value))
        d_tick = Decimal(str(tick))
        if d_tick <= 0:
            return float(d_value)
        rounded = (d_value / d_tick).to_integral_value(rounding=ROUND_HALF_UP) * d_tick
        return float(rounded)
    except (InvalidOperation, ValueError):
        return value


class RiskManager:
    def __init__(self, cfg, journal):
        self.cfg = cfg
        self.journal = journal

    def can_open(self, open_positions_count: int) -> tuple[bool, str]:
        """Check whether a new position may be opened. Reason is in Russian."""
        if open_positions_count >= self.cfg.max_positions:
            return False, (
                f"достигнут лимит открытых позиций ({self.cfg.max_positions})"
            )
        if self.daily_loss_breached():
            return False, "достигнут дневной лимит убытка, торговля остановлена"
        return True, ""

    def daily_loss_breached(self) -> bool:
        """True if today's realized PnL is at or below the daily loss limit."""
        try:
            pnl = self.journal.daily_realized_pnl()
        except Exception:
            logger.exception("failed to read daily realized pnl")
            return False
        limit = -self.cfg.daily_loss_limit * self.cfg.budget_usdt
        return pnl <= limit

    def calc_order(self, price: float, atr: float, side: str, instrument: dict) -> dict | None:
        """Compute qty/SL/TP for a new order, or None if the order is not viable.

        side: "Buy" (long) or "Sell" (short).
        instrument: {"qty_step": float, "min_qty": float, "tick_size": float}.
        """
        try:
            if price <= 0 or atr <= 0:
                logger.warning("calc_order: invalid price=%s atr=%s", price, atr)
                return None

            sl_dist = atr * self.cfg.atr_sl_mult
            if sl_dist <= 0:
                return None

            risk_usd = self.cfg.budget_usdt * self.cfg.risk_per_trade
            qty = risk_usd / sl_dist

            # Cap notional per position.
            max_notional = (
                self.cfg.budget_usdt * self.cfg.leverage / self.cfg.max_positions
            )
            if qty * price > max_notional:
                qty = max_notional / price

            qty = _floor_to_step(qty, instrument["qty_step"])
            if qty < instrument["min_qty"] or qty <= 0:
                logger.info(
                    "calc_order: qty %s below min_qty %s, skipping",
                    qty, instrument["min_qty"],
                )
                return None

            tp_dist = atr * self.cfg.atr_tp_mult
            if side == "Buy":
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                sl = price + sl_dist
                tp = price - tp_dist

            if sl <= 0 or tp <= 0:
                logger.warning("calc_order: non-positive SL/TP (sl=%s tp=%s)", sl, tp)
                return None

            tick = instrument["tick_size"]
            sl = _round_to_tick(sl, tick)
            tp = _round_to_tick(tp, tick)

            return {"qty": qty, "sl": sl, "tp": tp}
        except Exception:
            logger.exception("calc_order failed")
            return None
