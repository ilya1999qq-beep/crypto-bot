import time
from dataclasses import dataclass, field


@dataclass
class BotState:
    paused: bool = False
    pause_reason: str = ""
    cooldown_until: dict = field(default_factory=dict)   # symbol -> unix ts until which no new entries
    last_candle_ts: dict = field(default_factory=dict)   # symbol -> timestamp of last processed closed candle

    def in_cooldown(self, symbol: str) -> bool:
        return time.time() < self.cooldown_until.get(symbol, 0)

    def set_cooldown(self, symbol: str, seconds: float) -> None:
        self.cooldown_until[symbol] = time.time() + seconds
