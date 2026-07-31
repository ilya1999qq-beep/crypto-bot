# SPEC — Bybit Demo Multi-Agent Trading Bot

A crypto trading bot for Bybit **Demo Trading** (virtual funds, real market data), USDT linear
perpetuals, fully controlled from the phone via Telegram. Python 3.12, asyncio main loop.
pybit's HTTP client is synchronous — always call it via `asyncio.to_thread(...)` from async code.

## Global conventions (apply to every module)

- Secrets ONLY via `config.Config` (already written — read `config.py` before coding).
- **No network calls and no object construction with side effects at import time.** Everything is
  built inside `main()`.
- Logging: `logger = logging.getLogger("bot.<module>")`. Handlers are configured once in `main.py`
  (console + file `cfg.log_path`), modules just log.
- Code, identifiers and comments in English. All user-facing Telegram strings in **Russian**.
- All numeric parameters passed to pybit must be **strings**.
- Every module must survive bad data gracefully: wrap external calls, raise nothing at import.
- Already written, do not modify: `config.py`, `state.py`, `agents/base.py`, `agents/__init__.py`.

## Modules

### market_data.py
```python
class MarketData:
    def __init__(self, session): ...          # pybit.unified_trading.HTTP
    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame: ...
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame: ...
    def get_frames(self, symbol: str, cfg) -> tuple[pd.DataFrame, pd.DataFrame]: ...
        # (df15, df1h) both with indicators, using cfg.timeframe / cfg.trend_timeframe
```
- `get_klines`: `session.get_kline(category="linear", symbol=..., interval=..., limit=...)`.
  `result["list"]` is **newest-first** — reverse it. Columns: open, high, low, close, volume
  (floats), index = `pd.DatetimeIndex` from start-time ms (UTC). **Drop the last (still open)
  candle.**
- `add_indicators`: pure pandas, no external TA libraries. Adds columns:
  `ema20, ema50, ema200, rsi14, macd, macd_signal, macd_hist, atr14, bb_upper, bb_mid, bb_lower,
  adx14, obv`. Standard formulas (MACD 12/26/9, Bollinger 20/2, Wilder smoothing for RSI/ATR/ADX).

### agents/ — analysis agents
Each file defines one class subclassing `agents.base.BaseAgent`, implementing
`analyze(self, df15, df1h) -> AgentVote`. Score is an int in [-100, +100], positive = long.
`reason` = short Russian phrase (e.g. "восходящий тренд на 1ч, ADX 28"). Use only the last
~100 rows; guard against NaN (early rows) — if not enough data, return score 0.

- **agents/trend_agent.py — TrendAgent (name="trend")**: regime from 1h `ema50` vs `ema200`;
  direction from 15m `ema20` vs `ema50`; scale magnitude by 15m `adx14` (ADX<20 → weak, damp
  toward 0; ADX>40 → full strength). If 1h regime and 15m direction disagree, halve the score.
- **agents/momentum_agent.py — MomentumAgent (name="momentum")**: `rsi14` (<30 → bullish
  mean-revert bias, >70 → bearish; between — scaled distance from 50) combined with `macd_hist`
  sign and its slope over the last 3 bars.
- **agents/volume_agent.py — VolumeAgent (name="volume")**: OBV slope vs price slope over last
  ~20 bars — confirmation boosts score in price direction, divergence gives contrarian negative/
  positive tilt; last volume vs 20-bar average volume scales confidence.
- **agents/volatility_agent.py — VolatilityAgent (name="volatility")**: position of close within
  Bollinger bands + squeeze detection (band width vs its 50-bar mean). This agent mainly *damps*:
  if `atr14 > 2 *` its own 50-bar rolling mean → score 0, reason "экстремальная волатильность,
  лучше переждать". Otherwise small scores (|score| <= 40).
- **agents/llm_agent.py — LLMAgent (name="llm")**: constructor takes `cfg`. Inactive (always
  returns score 0, reason "LLM выключен") unless `cfg.llm_enabled and cfg.anthropic_api_key`.
  When active: `anthropic` SDK, model `"claude-haiku-4-5-20251001"`, `max_tokens=300`; prompt =
  compact text summary of the last 30 candles (OHLCV) + indicator snapshot; ask strictly for JSON
  `{"score": int, "reason": "..."}` (reason in Russian) and parse defensively. Cache the vote per
  symbol for 60 minutes (instance dict); method stays synchronous. Any exception →
  `AgentVote("llm", 0, "ошибка LLM")`. Import `anthropic` lazily inside the method.
  `analyze` here has signature `analyze(self, df15, df1h, symbol: str = "")` — extra optional arg.

### strategy.py
```python
@dataclass
class Decision:
    action: str            # "long" | "short" | "close" | "hold"
    score: float           # weighted ensemble score
    votes: list            # list[AgentVote]
    atr: float             # last atr14 of df15
    price: float           # last close of df15

class Strategy:
    def __init__(self, cfg): ...   # builds all agents
    def decide(self, symbol: str, df15, df1h, position_side: str | None) -> Decision: ...
```
- Weights: trend 0.30, momentum 0.30, volume 0.15, volatility 0.10, llm 0.15. If LLM disabled,
  renormalize remaining weights to sum 1.
- No position: score >= +cfg.signal_threshold → "long"; <= -threshold → "short"; else "hold".
- With position (`position_side` = "Buy"/"Sell"): if score crosses threshold **against** the
  position → "close", else "hold".
- `format_votes(votes) -> str` helper: multi-line Russian summary for Telegram
  (`"trend: +42 — восходящий тренд..."`).

### risk.py
```python
class RiskManager:
    def __init__(self, cfg, journal): ...
    def can_open(self, open_positions_count: int) -> tuple[bool, str]: ...
    def calc_order(self, price: float, atr: float, side: str, instrument: dict) -> dict | None: ...
    def daily_loss_breached(self) -> bool: ...
```
- `can_open`: False if `open_positions_count >= cfg.max_positions` or daily loss breached
  (reason string in Russian).
- `daily_loss_breached`: `journal.daily_realized_pnl() <= -cfg.daily_loss_limit * cfg.budget_usdt`.
- `calc_order`: `sl_dist = atr * cfg.atr_sl_mult`; `risk_usd = cfg.budget_usdt * cfg.risk_per_trade`;
  `qty = risk_usd / sl_dist`; cap notional: `qty * price <= cfg.budget_usdt * cfg.leverage /
  cfg.max_positions`. Round qty DOWN to `instrument["qty_step"]`; if `< instrument["min_qty"]` →
  return None. SL = price ∓ sl_dist, TP = price ± atr*cfg.atr_tp_mult (sign per side, "Buy" = long).
  Round SL/TP to `instrument["tick_size"]`. Return `{"qty": float, "sl": float, "tp": float}`.

### trader.py
```python
class Trader:
    def __init__(self, session, cfg): ...
    def setup(self) -> None: ...
    def instrument(self, symbol: str) -> dict: ...   # {"qty_step","min_qty","tick_size"} floats
    def get_positions(self) -> dict: ...
    def open_position(self, symbol, side, qty, sl, tp) -> dict: ...
    def close_position(self, symbol: str) -> None: ...
    def get_equity(self) -> float: ...
    def last_closed_pnl(self, symbol: str) -> float | None: ...
```
- `setup`: for each cfg.symbol — `set_leverage(category="linear", symbol=..., buyLeverage=str(cfg.leverage),
  sellLeverage=...)`, **swallow error "leverage not modified" (ErrCode 110043)**; cache instrument
  info via `get_instruments_info(category="linear", symbol=...)`: `lotSizeFilter.qtyStep`,
  `lotSizeFilter.minOrderQty`, `priceFilter.tickSize`.
- `get_positions`: `get_positions(category="linear", settleCoin="USDT")`, only entries with
  `size > 0` → `{symbol: {"side": "Buy"|"Sell", "qty": float, "entry": float, "unrealized": float}}`.
- `open_position`: `place_order(category="linear", symbol=..., side=..., orderType="Market",
  qty=str(...), takeProfit=str(...), stopLoss=str(...), positionIdx=0)`.
- `close_position`: market order, opposite side, full size, `reduceOnly=True`, positionIdx=0.
- `get_equity`: `get_wallet_balance(accountType="UNIFIED")` → `totalEquity` as float.
- `last_closed_pnl`: `get_closed_pnl(category="linear", symbol=..., limit=1)` → `closedPnl` float,
  None if empty.

### journal.py
```python
class Journal:
    def __init__(self, db_path: str): ...   # sqlite3, check_same_thread=False + threading.Lock
    def log_open(self, symbol, side, qty, entry, sl, tp, reason: str) -> None: ...
    def log_close(self, symbol, exit_price: float | None, pnl: float | None, reason: str) -> None: ...
    def has_open_trade(self, symbol: str) -> bool: ...
    def open_symbols(self) -> list[str]: ...
    def daily_realized_pnl(self) -> float: ...   # sum of pnl for trades closed today (UTC)
    def snapshot_equity(self, equity: float) -> None: ...
    def report_text(self) -> str: ...   # Russian: last 10 closed trades, totals, winrate
```
Tables: `trades(id, ts_open, ts_close, symbol, side, qty, entry, exit, sl, tp, pnl, reason_open,
reason_close, status)` (`status` = 'open'/'closed'), `equity(ts, value)`. `log_close` closes the
latest open trade for the symbol.

### telegram_bot.py
aiogram 3.x.
```python
def build_bot(cfg, deps: dict) -> tuple[Bot, Dispatcher]: ...
async def notify(bot, cfg, text: str) -> None: ...   # send_message to cfg.telegram_chat_id, never raises
```
`deps`: `{"state": BotState, "trader": Trader, "journal": Journal, "cfg": Config}`.
**Ignore all updates whose `message.chat.id != cfg.telegram_chat_id`** (silent drop).
Commands (all replies in Russian, use emoji moderately):
- `/start`, `/help` — command list
- `/status` — paused or running (+reason), open positions with uPnL, today's realized PnL, equity
- `/balance` — demo equity + configured bot budget
- `/positions` — detailed open positions
- `/pause` — pause trading (positions stay open); `/resume` — resume, clear pause_reason
- `/close BTCUSDT` or `/close all` — close position(s) via trader (call via `asyncio.to_thread`)
- `/report` — `journal.report_text()`
All trader/journal calls from handlers go through `asyncio.to_thread`. Handlers must catch
exceptions and reply with a readable Russian error instead of crashing.

### main.py (integration)
- `setup_logging(cfg)`: console + file handlers on logger "bot".
- Validate required env (bybit keys, telegram token, chat id) — if missing, print a clear Russian
  message listing exactly what's missing in `.env` and exit(1).
- `session = HTTP(demo=True, api_key=cfg.bybit_api_key, api_secret=cfg.bybit_api_secret)`
  (`from pybit.unified_trading import HTTP`).
- Build MarketData, Strategy, RiskManager, Journal, Trader (`await asyncio.to_thread(trader.setup)`),
  BotState; `bot, dp = build_bot(cfg, deps)`.
- On start: notify "🤖 Бот запущен (Bybit DEMO)" + short config summary (pairs, leverage, budget).
- `await asyncio.gather(dp.start_polling(bot), trade_loop(...))`.
- `trade_loop`, every `cfg.poll_seconds`:
  1. If `state.paused` → sleep, continue.
  2. `positions = await asyncio.to_thread(trader.get_positions)`.
  3. **Reconcile**: for each symbol in `journal.open_symbols()` that is NOT in positions — it was
     closed by SL/TP on the exchange: `pnl = trader.last_closed_pnl(symbol)`,
     `journal.log_close(symbol, None, pnl, "SL/TP на бирже")`, notify, set cooldown
     (`cfg.cooldown_candles * int(cfg.timeframe) * 60` seconds).
  4. Daily-loss check: if breached and not paused → `state.paused = True`, pause_reason, notify once.
  5. For each symbol: `df15, df1h = await asyncio.to_thread(md.get_frames, symbol, cfg)`;
     skip entries if `df15.index[-1] == state.last_candle_ts.get(symbol)` (no new closed candle) —
     but still evaluate "close" for open positions every poll; update `last_candle_ts`.
     `decision = strategy.decide(symbol, df15, df1h, position side or None)`.
  6. "long"/"short": skip if symbol already has a position, is in cooldown, or `can_open` fails;
     `order = risk.calc_order(...)`; if None → log & skip; `trader.open_position(...)`;
     `journal.log_open(...)`; notify with price, qty, SL/TP and `format_votes`.
  7. "close": `trader.close_position`; `pnl = trader.last_closed_pnl`; `journal.log_close(...,
     "сигнал развернулся")`; notify; cooldown.
  8. Every ~15 min: `journal.snapshot_equity(trader.get_equity())`.
  9. Whole loop body in try/except: log exceptions; notify at most once per 10 minutes.
- `if __name__ == "__main__": asyncio.run(main())` with KeyboardInterrupt handled cleanly.
