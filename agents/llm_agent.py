"""LLM analysis agent backed by the Anthropic API (Claude Haiku).

Inactive unless cfg.llm_enabled and cfg.anthropic_api_key are set.
The `anthropic` package is imported lazily inside the method so the module
can be imported (and the whole bot can run) without the SDK installed.
"""

import json
import logging
import re
import time

from agents.base import AgentVote, BaseAgent

logger = logging.getLogger("bot.llm_agent")

MODEL = "claude-haiku-4-5-20251001"
CACHE_TTL_SECONDS = 60 * 60  # cache one vote per symbol for 60 minutes

INDICATOR_COLS = (
    "ema20",
    "ema50",
    "ema200",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr14",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "adx14",
    "obv",
)


class LLMAgent(BaseAgent):
    name = "llm"

    def __init__(self, cfg):
        self.cfg = cfg
        # symbol -> (unix_ts, AgentVote)
        self._cache: dict = {}
        self._client = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def analyze(self, df15, df1h, symbol: str = "") -> AgentVote:
        if not (self.cfg.llm_enabled and self.cfg.anthropic_api_key):
            return AgentVote(self.name, 0, "LLM выключен")

        cached = self._cache.get(symbol)
        if cached is not None:
            ts, vote = cached
            if time.time() - ts < CACHE_TTL_SECONDS:
                return vote

        try:
            vote = self._query_llm(df15, df1h, symbol)
        except Exception:
            logger.exception("LLM query failed for %s", symbol)
            return AgentVote(self.name, 0, "ошибка LLM")

        self._cache[symbol] = (time.time(), vote)
        return vote

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _get_client(self):
        # Lazy import: no SDK dependency (and no client construction) at import time.
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)
        return self._client

    def _query_llm(self, df15, df1h, symbol: str) -> AgentVote:
        prompt = self._build_prompt(df15, df1h, symbol)
        client = self._get_client()

        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=(
                "You are a crypto trading analyst. Analyze the provided candle and "
                "indicator data and respond with STRICTLY one JSON object and nothing "
                'else: {"score": <int from -100 to 100, positive = long bias>, '
                '"reason": "<short phrase in Russian, max 80 chars>"}'
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        score, reason = self._parse_response(text)
        return AgentVote(self.name, score, reason)

    def _build_prompt(self, df15, df1h, symbol: str) -> str:
        lines = [f"Symbol: {symbol or 'unknown'}"]

        lines.append("Last 30 candles, 15m timeframe (time O/H/L/C/V):")
        tail = df15.tail(30)
        for ts, row in tail.iterrows():
            lines.append(
                "%s %.6g/%.6g/%.6g/%.6g/%.6g"
                % (
                    ts.strftime("%m-%d %H:%M"),
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                )
            )

        lines.append("Latest 15m indicators:")
        lines.append(self._indicator_snapshot(df15))
        lines.append("Latest 1h indicators:")
        lines.append(self._indicator_snapshot(df1h))
        lines.append(
            'Reply with ONLY the JSON object {"score": int, "reason": "..."} '
            "(reason in Russian)."
        )
        return "\n".join(lines)

    @staticmethod
    def _indicator_snapshot(df) -> str:
        parts = []
        last = df.iloc[-1]
        for col in INDICATOR_COLS:
            if col in df.columns:
                try:
                    value = float(last[col])
                except (TypeError, ValueError):
                    continue
                if value == value:  # skip NaN
                    parts.append("%s=%.6g" % (col, value))
        return " ".join(parts) if parts else "n/a"

    def _parse_response(self, text: str) -> tuple[int, str]:
        """Defensively extract {"score": int, "reason": str} from model output."""
        data = None
        try:
            data = json.loads(text.strip())
        except (ValueError, TypeError):
            # Model may wrap JSON in prose or code fences — grab the first {...} block.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except (ValueError, TypeError):
                    data = None

        if not isinstance(data, dict):
            return 0, "ошибка LLM"

        try:
            score = int(float(data.get("score", 0)))
        except (TypeError, ValueError):
            score = 0
        score = max(-100, min(100, score))

        reason = data.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            reason = "без комментария"
        return score, reason.strip()[:200]
