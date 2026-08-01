import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Config:
    # --- secrets: env only, never hardcode ---
    bybit_api_key: str = _env("BYBIT_API_KEY")
    bybit_api_secret: str = _env("BYBIT_API_SECRET")
    binance_api_key: str = _env("BINANCE_API_KEY")
    binance_api_secret: str = _env("BINANCE_API_SECRET")
    okx_api_key: str = _env("OKX_API_KEY")
    okx_api_secret: str = _env("OKX_API_SECRET")
    okx_passphrase: str = _env("OKX_PASSPHRASE")
    telegram_token: str = _env("TELEGRAM_TOKEN")
    telegram_chat_id: int = int(_env("TELEGRAM_CHAT_ID", "0") or 0)
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY")
    llm_enabled: bool = _env("LLM_ENABLED", "0").lower() in ("1", "true", "yes")

    # --- trading ---
    # "paper"   = real public prices, simulated fills, no exchange account
    # "bybit"   = Bybit Demo (geo-blocked from US IPs -> unusable on GitHub Actions)
    # "binance" = Binance USDT-M Futures Testnet (needs a Binance account)
    # "okx"     = OKX Demo Trading (needs identity verification)
    exchange: str = _env("EXCHANGE", "bybit").lower()
    demo: bool = True                      # always a paper/demo environment
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    leverage: int = 3
    budget_usdt: float = 300.0             # virtual budget the bot is allowed to manage
    risk_per_trade: float = 0.02           # 2% of budget risked per trade (loss at SL)
    max_positions: int = 2                 # concurrent positions across all symbols
    daily_loss_limit: float = 0.06         # -6% of budget realized in a day -> auto-pause
    timeframe: str = "15"                  # primary candles, minutes
    trend_timeframe: str = "60"            # higher-timeframe trend filter
    poll_seconds: int = 60
    signal_threshold: float = 35.0         # |ensemble score| needed to act, 0..100
    atr_sl_mult: float = 1.5               # stop-loss distance in ATRs
    atr_tp_mult: float = 2.5               # take-profit distance in ATRs
    cooldown_candles: int = 3              # candles to wait on a symbol after closing a trade

    # --- infra ---
    db_path: str = "bot.db"
    log_path: str = "bot.log"
