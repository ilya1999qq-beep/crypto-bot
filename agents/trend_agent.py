"""Trend agent: multi-timeframe EMA regime/direction scaled by ADX strength."""

import logging
import math

from agents.base import AgentVote, BaseAgent

logger = logging.getLogger("bot.trend_agent")

# ADX thresholds: below ADX_WEAK the trend is considered weak (score damped
# toward 0), above ADX_STRONG the trend is at full strength.
ADX_WEAK = 20.0
ADX_STRONG = 40.0
WEAK_FLOOR_FACTOR = 0.25  # max fraction of full strength when ADX < ADX_WEAK


def _last_valid(df, columns):
    """Return dict of last-row values for columns, or None if missing/NaN."""
    if df is None or len(df) == 0:
        return None
    row = df.iloc[-1]
    out = {}
    for col in columns:
        if col not in df.columns:
            return None
        val = row[col]
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        out[col] = val
    return out


class TrendAgent(BaseAgent):
    name = "trend"

    def analyze(self, df15, df1h) -> AgentVote:
        try:
            # Use only the recent part of the frames.
            if df15 is not None and len(df15) > 100:
                df15 = df15.tail(100)
            if df1h is not None and len(df1h) > 100:
                df1h = df1h.tail(100)

            h = _last_valid(df1h, ["ema50", "ema200"])
            m = _last_valid(df15, ["ema20", "ema50", "adx14"])
            if h is None or m is None:
                return AgentVote(self.name, 0, "недостаточно данных для тренда")

            # 1h regime: bull if ema50 above ema200.
            regime = 1 if h["ema50"] > h["ema200"] else -1
            # 15m direction: bull if ema20 above ema50.
            direction = 1 if m["ema20"] > m["ema50"] else -1

            adx = m["adx14"]
            # Scale magnitude by ADX: <20 -> damped toward 0, >40 -> full.
            if adx >= ADX_STRONG:
                strength = 1.0
            elif adx <= ADX_WEAK:
                strength = max(adx, 0.0) / ADX_WEAK * WEAK_FLOOR_FACTOR
            else:
                strength = WEAK_FLOOR_FACTOR + (adx - ADX_WEAK) / (
                    ADX_STRONG - ADX_WEAK
                ) * (1.0 - WEAK_FLOOR_FACTOR)

            score = direction * 100.0 * strength

            aligned = regime == direction
            if not aligned:
                # Higher timeframe disagrees with local direction -> halve.
                score /= 2.0

            score_int = int(max(-100, min(100, round(score))))

            regime_txt = "восходящий" if regime > 0 else "нисходящий"
            dir_txt = "вверх" if direction > 0 else "вниз"
            if aligned:
                reason = f"{regime_txt} тренд на 1ч, 15м {dir_txt}, ADX {adx:.0f}"
            else:
                reason = (
                    f"{regime_txt} тренд на 1ч против 15м ({dir_txt}), "
                    f"ADX {adx:.0f}"
                )
            if adx < ADX_WEAK:
                reason += ", слабый тренд"

            return AgentVote(self.name, score_int, reason)
        except Exception:
            logger.exception("TrendAgent failed")
            return AgentVote(self.name, 0, "ошибка анализа тренда")
