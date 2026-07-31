"""Ensemble strategy: combines agent votes into a trading decision."""

import logging
import math
from dataclasses import dataclass

from agents.base import AgentVote
from agents.trend_agent import TrendAgent
from agents.momentum_agent import MomentumAgent
from agents.volume_agent import VolumeAgent
from agents.volatility_agent import VolatilityAgent
from agents.llm_agent import LLMAgent

logger = logging.getLogger("bot.strategy")

# Base ensemble weights per agent name.
WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.30,
    "volume": 0.15,
    "volatility": 0.10,
    "llm": 0.15,
}


@dataclass
class Decision:
    action: str            # "long" | "short" | "close" | "hold"
    score: float           # weighted ensemble score
    votes: list            # list[AgentVote]
    atr: float             # last atr14 of df15
    price: float           # last close of df15


def format_votes(votes: list) -> str:
    """Multi-line Russian summary of agent votes for Telegram."""
    lines = []
    for v in votes:
        lines.append(f"{v.name}: {v.score:+d} — {v.reason}")
    return "\n".join(lines)


class Strategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trend = TrendAgent()
        self.momentum = MomentumAgent()
        self.volume = VolumeAgent()
        self.volatility = VolatilityAgent()
        self.llm = LLMAgent(cfg)
        self.llm_active = bool(cfg.llm_enabled and cfg.anthropic_api_key)

        # Effective weights: drop LLM and renormalize when it is disabled.
        weights = dict(WEIGHTS)
        if not self.llm_active:
            weights.pop("llm")
            total = sum(weights.values())
            weights = {k: w / total for k, w in weights.items()}
        self.weights = weights

    def _collect_votes(self, symbol: str, df15, df1h) -> list:
        votes = []
        plain_agents = [self.trend, self.momentum, self.volume, self.volatility]
        for agent in plain_agents:
            try:
                votes.append(agent.analyze(df15, df1h))
            except Exception:
                logger.exception("agent %s failed", agent.name)
                votes.append(AgentVote(agent.name, 0, "ошибка агента"))
        if self.llm_active:
            try:
                votes.append(self.llm.analyze(df15, df1h, symbol))
            except Exception:
                logger.exception("agent llm failed")
                votes.append(AgentVote("llm", 0, "ошибка LLM"))
        return votes

    def decide(self, symbol: str, df15, df1h, position_side: str | None) -> Decision:
        votes = self._collect_votes(symbol, df15, df1h)

        score = 0.0
        for v in votes:
            score += self.weights.get(v.name, 0.0) * v.score

        # Last closed-candle values; guard against missing/NaN data.
        atr = 0.0
        price = 0.0
        try:
            atr_val = float(df15["atr14"].iloc[-1])
            price_val = float(df15["close"].iloc[-1])
            if not math.isnan(atr_val):
                atr = atr_val
            if not math.isnan(price_val):
                price = price_val
        except Exception:
            logger.exception("failed to read atr/price for %s", symbol)

        threshold = self.cfg.signal_threshold
        if position_side is None:
            if score >= threshold:
                action = "long"
            elif score <= -threshold:
                action = "short"
            else:
                action = "hold"
        else:
            # Close only when the signal crosses threshold against the position.
            if position_side == "Buy" and score <= -threshold:
                action = "close"
            elif position_side == "Sell" and score >= threshold:
                action = "close"
            else:
                action = "hold"

        logger.info(
            "%s: score=%.1f action=%s position=%s", symbol, score, action, position_side
        )
        return Decision(action=action, score=score, votes=votes, atr=atr, price=price)
