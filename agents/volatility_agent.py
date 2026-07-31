"""Volatility agent: Bollinger-band position + squeeze detection.

Mainly a damping agent: it vetoes trading in extreme volatility and
otherwise contributes only small scores (|score| <= 40).
"""

import logging

from agents.base import AgentVote, BaseAgent

logger = logging.getLogger("bot.volatility_agent")

MAX_SCORE = 40          # this agent never shouts louder than this
ROLLING_WINDOW = 50     # bars for ATR / band-width rolling means
SQUEEZE_RATIO = 0.75    # band width below 75% of its mean -> squeeze


class VolatilityAgent(BaseAgent):
    name = "volatility"

    def analyze(self, df15, df1h) -> AgentVote:
        try:
            return self._analyze(df15)
        except Exception:
            logger.exception("volatility agent failed")
            return AgentVote(self.name, 0, "ошибка анализа волатильности")

    def _analyze(self, df15) -> AgentVote:
        needed = ["close", "atr14", "bb_upper", "bb_mid", "bb_lower"]
        if df15 is None or len(df15) < ROLLING_WINDOW + 20:
            return AgentVote(self.name, 0, "недостаточно данных")
        for col in needed:
            if col not in df15.columns:
                return AgentVote(self.name, 0, "недостаточно данных")

        df = df15.tail(100).copy()
        tail = df[needed].dropna()
        if len(tail) < ROLLING_WINDOW + 5:
            return AgentVote(self.name, 0, "недостаточно данных")

        close = float(tail["close"].iloc[-1])
        atr = float(tail["atr14"].iloc[-1])
        bb_upper = float(tail["bb_upper"].iloc[-1])
        bb_mid = float(tail["bb_mid"].iloc[-1])
        bb_lower = float(tail["bb_lower"].iloc[-1])

        # --- extreme volatility veto -------------------------------------
        atr_mean = float(tail["atr14"].rolling(ROLLING_WINDOW).mean().iloc[-1])
        if atr_mean > 0 and atr > 2.0 * atr_mean:
            return AgentVote(
                self.name, 0, "экстремальная волатильность, лучше переждать"
            )

        # --- position of close inside the bands, in [-1, +1] -------------
        half_width = (bb_upper - bb_lower) / 2.0
        if half_width <= 0:
            return AgentVote(self.name, 0, "полосы Боллинджера вырождены")
        pos = (close - bb_mid) / half_width
        pos = max(-1.0, min(1.0, pos))

        # --- squeeze detection: band width vs its 50-bar mean ------------
        width = tail["bb_upper"] - tail["bb_lower"]
        width_mean = float(width.rolling(ROLLING_WINDOW).mean().iloc[-1])
        squeeze = width_mean > 0 and float(width.iloc[-1]) < SQUEEZE_RATIO * width_mean

        if squeeze:
            # Narrow bands: energy is building, mildly favor the side of the
            # mid-band the price is on (likely breakout direction).
            score = int(round(pos * 25))
            reason = "сжатие полос Боллинджера, возможен импульс"
            if abs(pos) < 0.15:
                score = 0
                reason = "сжатие полос, цена у середины — ждём импульса"
        else:
            # Normal regime: mild mean-reversion bias — price near the upper
            # band is stretched long (bearish tilt) and vice versa.
            score = int(round(-pos * 30))
            if pos > 0.8:
                reason = "цена у верхней полосы Боллинджера, перегрев"
            elif pos < -0.8:
                reason = "цена у нижней полосы Боллинджера, перепроданность"
            else:
                reason = "волатильность в норме, цена внутри полос"

        score = max(-MAX_SCORE, min(MAX_SCORE, score))
        return AgentVote(self.name, score, reason)
