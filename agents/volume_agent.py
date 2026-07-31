"""Volume analysis agent: OBV/price slope confirmation and divergence."""

import logging

import numpy as np

from agents.base import AgentVote, BaseAgent

logger = logging.getLogger("bot.volume_agent")

LOOKBACK = 20  # bars used for slope / average-volume calculations


def _norm_slope(series) -> float:
    """Linear-regression slope of the series, normalized by its mean absolute level.

    Returns a dimensionless slope so price and OBV slopes are comparable.
    """
    y = np.asarray(series, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    scale = np.mean(np.abs(y))
    if scale == 0 or not np.isfinite(scale):
        return 0.0
    return float(slope / scale)


class VolumeAgent(BaseAgent):
    name = "volume"

    def analyze(self, df15, df1h) -> AgentVote:
        try:
            df = df15.tail(100)
            required = ("close", "volume", "obv")
            if any(c not in df.columns for c in required):
                return AgentVote(self.name, 0, "нет данных по объёму")

            window = df[list(required)].dropna().tail(LOOKBACK)
            if len(window) < LOOKBACK:
                return AgentVote(self.name, 0, "недостаточно данных")

            price_slope = _norm_slope(window["close"])
            obv_slope = _norm_slope(window["obv"])

            # Confidence multiplier from last volume vs 20-bar average volume.
            avg_vol = float(window["volume"].mean())
            last_vol = float(window["volume"].iloc[-1])
            if avg_vol <= 0:
                vol_ratio = 1.0
            else:
                vol_ratio = last_vol / avg_vol
            # Clamp so a single spike/drought does not dominate the score.
            vol_factor = min(max(vol_ratio, 0.5), 1.5)

            price_dir = 1 if price_slope > 0 else (-1 if price_slope < 0 else 0)
            obv_dir = 1 if obv_slope > 0 else (-1 if obv_slope < 0 else 0)

            # Base magnitude from how pronounced both slopes are (dimensionless,
            # typical normalized slopes are small — amplify and cap at 1).
            strength = min(1.0, (abs(price_slope) + abs(obv_slope)) * 0.5 * 2000.0)

            if price_dir == 0 or obv_dir == 0:
                return AgentVote(self.name, 0, "объём без направления")

            if price_dir == obv_dir:
                # OBV confirms the price move -> score in price direction.
                score = int(round(price_dir * 70 * strength * vol_factor))
                trend_word = "рост" if price_dir > 0 else "снижение"
                reason = f"объём подтверждает {trend_word} (OBV в ту же сторону)"
            else:
                # Divergence -> contrarian tilt against the price direction.
                score = int(round(-price_dir * 40 * strength * vol_factor))
                reason = "дивергенция OBV и цены, возможен разворот"

            score = max(-100, min(100, score))
            if vol_ratio < 0.7:
                reason += ", объём ниже среднего"
            elif vol_ratio > 1.3:
                reason += ", объём выше среднего"

            return AgentVote(self.name, score, reason)
        except Exception:
            logger.exception("volume agent failed")
            return AgentVote(self.name, 0, "ошибка анализа объёма")
