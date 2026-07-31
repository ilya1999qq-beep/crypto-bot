from dataclasses import dataclass


@dataclass
class AgentVote:
    name: str
    score: int   # -100..+100, positive = long bias, negative = short bias
    reason: str  # short human-readable phrase in Russian, shown in Telegram reports


class BaseAgent:
    name = "base"

    def analyze(self, df15, df1h) -> AgentVote:
        """df15/df1h: pandas DataFrames with OHLCV + indicators (see market_data.add_indicators)."""
        raise NotImplementedError
