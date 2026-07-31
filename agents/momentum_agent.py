"""Momentum agent: RSI level combined with MACD histogram sign and slope."""

import logging

import pandas as pd

from agents.base import AgentVote, BaseAgent

logger = logging.getLogger("bot.momentum_agent")


class MomentumAgent(BaseAgent):
    name = "momentum"

    def analyze(self, df15: pd.DataFrame, df1h: pd.DataFrame) -> AgentVote:
        try:
            df = df15.tail(100)
            if len(df) < 5:
                return AgentVote(self.name, 0, "недостаточно данных")

            rsi = df["rsi14"].iloc[-1]
            hist = df["macd_hist"].iloc[-3:]

            if pd.isna(rsi) or hist.isna().any():
                return AgentVote(self.name, 0, "недостаточно данных")

            # --- RSI component ---
            # <30 -> bullish mean-revert bias, >70 -> bearish,
            # in between -> scaled distance from 50 (momentum direction).
            if rsi < 30:
                rsi_score = min(100.0, 50.0 + (30.0 - rsi) * 2.5)
                rsi_desc = f"RSI {rsi:.0f} — перепроданность"
            elif rsi > 70:
                rsi_score = -min(100.0, 50.0 + (rsi - 70.0) * 2.5)
                rsi_desc = f"RSI {rsi:.0f} — перекупленность"
            else:
                rsi_score = (rsi - 50.0) / 20.0 * 60.0  # +-60 at RSI 70/30
                rsi_desc = f"RSI {rsi:.0f}"

            # --- MACD histogram component: sign + slope over last 3 bars ---
            h0, h1, h2 = float(hist.iloc[-3]), float(hist.iloc[-2]), float(hist.iloc[-1])
            slope = (h2 - h0) / 2.0

            # Normalize histogram magnitude by recent price scale to stay unitless.
            price = float(df["close"].iloc[-1])
            scale = max(abs(price) * 1e-4, 1e-12)  # ~1 bp of price as unit

            sign_score = max(-50.0, min(50.0, (h2 / scale) * 10.0))
            slope_score = max(-50.0, min(50.0, (slope / scale) * 20.0))
            macd_score = sign_score + slope_score

            if h2 > 0 and slope > 0:
                macd_desc = "MACD растёт выше нуля"
            elif h2 > 0:
                macd_desc = "MACD выше нуля, но слабеет"
            elif h2 < 0 and slope < 0:
                macd_desc = "MACD падает ниже нуля"
            elif h2 < 0:
                macd_desc = "MACD ниже нуля, но разворачивается"
            else:
                macd_desc = "MACD нейтрален"

            score = int(round(0.5 * rsi_score + 0.5 * macd_score))
            score = max(-100, min(100, score))

            return AgentVote(self.name, score, f"{rsi_desc}, {macd_desc}")
        except Exception:
            logger.exception("momentum agent failed")
            return AgentVote(self.name, 0, "ошибка анализа моментума")
