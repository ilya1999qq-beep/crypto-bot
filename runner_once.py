"""Single-pass runner for serverless (GitHub Actions cron) mode.

Instead of a long-running loop, this script does ONE full pass and exits:
  1. process pending Telegram commands (getUpdates since the stored offset),
  2. reconcile exchange-closed positions (SL/TP hits between runs),
  3. run the agent ensemble per symbol and open/close positions,
  4. snapshot equity.

All cross-run state (paused flag, telegram offset, last processed candle,
cooldowns) lives in the sqlite `state` table, so the bot.db file committed
back to the repo carries full continuity between runs. SL/TP orders sit on
the exchange itself, so positions stay protected while nothing is running.

Everything here is synchronous — no asyncio, no aiogram. Telegram is driven
directly over the HTTP Bot API via `requests` (a pybit dependency).
"""

import logging
import sys
import time
from datetime import datetime, timezone

import requests

from config import Config
from journal import Journal
from market_data import MarketData
from risk import RiskManager
from strategy import Strategy, format_votes
from trader import Trader

logger = logging.getLogger("bot.runner")

TG_API = "https://api.telegram.org/bot{token}/{method}"

HELP_TEXT = (
    "Команды:\n"
    "/status — состояние и позиции\n"
    "/balance — демо-баланс\n"
    "/positions — открытые позиции\n"
    "/pause — приостановить торговлю\n"
    "/resume — возобновить\n"
    "/close BTCUSDT или /close all — закрыть позицию(и)\n"
    "/report — журнал сделок\n\n"
    "⏱ Бот работает по расписанию (раз в ~15 мин), команды применяются "
    "при следующем запуске."
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------- telegram

def tg_send(cfg, text: str) -> None:
    """Send a plain-text message to the owner; never raises."""
    try:
        requests.post(
            TG_API.format(token=cfg.telegram_token, method="sendMessage"),
            json={"chat_id": cfg.telegram_chat_id, "text": text},
            timeout=15,
        )
    except Exception:
        logger.exception("tg_send failed")


def tg_get_updates(cfg, offset: int) -> list:
    try:
        resp = requests.get(
            TG_API.format(token=cfg.telegram_token, method="getUpdates"),
            params={"offset": offset, "timeout": 0},
            timeout=15,
        )
        data = resp.json()
        return data.get("result", []) if data.get("ok") else []
    except Exception:
        logger.exception("tg_get_updates failed")
        return []


def _fmt_positions(positions: dict) -> str:
    if not positions:
        return "Открытых позиций нет."
    lines = []
    for sym, p in positions.items():
        side_ru = "лонг 📈" if p["side"] == "Buy" else "шорт 📉"
        upnl = p.get("unrealized", 0.0)
        sign = "+" if upnl >= 0 else ""
        lines.append(
            f"• {sym}: {side_ru}, объём {p['qty']}, вход {p['entry']}, "
            f"uPnL {sign}{upnl:.2f} USDT"
        )
    return "\n".join(lines)


def process_commands(cfg, journal, trader) -> None:
    """Apply Telegram commands accumulated since the previous run."""
    offset = int(journal.get_state("tg_offset", "0") or 0)
    updates = tg_get_updates(cfg, offset)
    if not updates:
        return

    positions_cache: dict | None = None
    for upd in updates:
        offset = max(offset, upd.get("update_id", 0) + 1)
        msg = upd.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        text = (msg.get("text") or "").strip()
        if chat_id != cfg.telegram_chat_id or not text.startswith("/"):
            continue
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower().split("@")[0]
        arg = arg.strip()
        try:
            if cmd in ("/start", "/help"):
                tg_send(cfg, HELP_TEXT)

            elif cmd == "/pause":
                journal.set_state("paused", "1")
                journal.set_state("pause_reason", "пауза вручную")
                tg_send(cfg, "⏸ Торговля приостановлена. /resume — возобновить.")

            elif cmd == "/resume":
                journal.set_state("paused", "0")
                journal.set_state("pause_reason", "")
                tg_send(cfg, "▶️ Торговля возобновлена.")

            elif cmd in ("/status", "/positions", "/balance"):
                if positions_cache is None:
                    positions_cache = trader.get_positions()
                if cmd == "/balance":
                    equity = trader.get_equity()
                    tg_send(
                        cfg,
                        f"💰 Демо-баланс: {equity:.2f} USDT\n"
                        f"Бюджет бота: {cfg.budget_usdt:.0f} USDT",
                    )
                elif cmd == "/positions":
                    tg_send(cfg, _fmt_positions(positions_cache))
                else:
                    paused = journal.get_state("paused", "0") == "1"
                    reason = journal.get_state("pause_reason", "") or ""
                    state_ru = "⏸ пауза" + (f" ({reason})" if reason else "") \
                        if paused else "▶️ работает"
                    daily = journal.daily_realized_pnl()
                    sign = "+" if daily >= 0 else ""
                    tg_send(
                        cfg,
                        f"Состояние: {state_ru}\n"
                        f"PnL за сегодня: {sign}{daily:.2f} USDT\n\n"
                        + _fmt_positions(positions_cache),
                    )

            elif cmd == "/close":
                if positions_cache is None:
                    positions_cache = trader.get_positions()
                targets = (
                    list(positions_cache) if arg.lower() in ("all", "все")
                    else [arg.upper()] if arg else []
                )
                if not targets:
                    tg_send(cfg, "Укажи символ: /close BTCUSDT или /close all")
                    continue
                for sym in targets:
                    close_since = int(time.time() * 1000) - 10_000
                    if trader.close_position(sym):
                        pnl = trader.last_closed_pnl(sym, close_since)
                        journal.log_close(
                            sym, None, pnl, "закрыто вручную через Telegram"
                        )
                        positions_cache.pop(sym, None)
                        pnl_txt = (
                            f"{'+' if pnl >= 0 else ''}{pnl:.2f} USDT"
                            if pnl is not None else "н/д"
                        )
                        tg_send(cfg, f"✅ {sym}: позиция закрыта, PnL {pnl_txt}")
                    else:
                        tg_send(cfg, f"У {sym} нет открытой позиции.")

            elif cmd == "/report":
                tg_send(cfg, journal.report_text())

        except Exception as exc:
            logger.exception("command %s failed", cmd)
            tg_send(cfg, f"⚠️ Ошибка при выполнении {cmd}: {exc}")

    journal.set_state("tg_offset", str(offset))


# ------------------------------------------------------------- trading pass

def _cooldown_seconds(cfg) -> float:
    try:
        return cfg.cooldown_candles * int(cfg.timeframe) * 60
    except (TypeError, ValueError):
        return cfg.cooldown_candles * 15 * 60


def in_cooldown(journal, symbol: str) -> bool:
    until = float(journal.get_state(f"cooldown_until:{symbol}", "0") or 0)
    return time.time() < until


def set_cooldown(cfg, journal, symbol: str) -> None:
    journal.set_state(
        f"cooldown_until:{symbol}", str(time.time() + _cooldown_seconds(cfg))
    )


def trading_pass(cfg, journal, md, strategy, risk, trader) -> None:
    if journal.get_state("paused", "0") == "1":
        logger.info("bot is paused, skipping trading pass")
        return

    # Raises on API failure -> the whole pass is aborted by the caller;
    # never reconcile or open entries off an error-empty snapshot.
    positions = trader.get_positions()

    # --- reconcile: SL/TP hits that happened between runs
    for symbol in journal.open_symbols():
        if symbol in positions:
            continue
        trader.cancel_stray_orders(symbol)  # Binance: remove the surviving SL/TP twin
        since_ms = journal.open_trade_since_ms(symbol)
        pnl = trader.last_closed_pnl(symbol, since_ms)
        journal.log_close(symbol, None, pnl, "сработал SL или TP")
        set_cooldown(cfg, journal, symbol)
        pnl_txt = (
            f"{'+' if pnl >= 0 else ''}{pnl:.2f} USDT" if pnl is not None else "н/д"
        )
        tg_send(cfg, f"🎯 {symbol}: позиция закрыта по SL/TP, PnL {pnl_txt}")
        logger.info("reconciled exchange-closed position %s pnl=%s", symbol, pnl)

    # --- daily loss guard
    if risk.daily_loss_breached():
        journal.set_state("paused", "1")
        journal.set_state("pause_reason", "достигнут дневной лимит убытка")
        tg_send(
            cfg,
            "🛑 Достигнут дневной лимит убытка — торговля автоматически "
            "приостановлена. /resume — возобновить вручную.",
        )
        logger.warning("daily loss limit breached, bot paused")
        return

    # --- per-symbol analysis (one failing symbol must not starve the others)
    for symbol in cfg.symbols:
        try:
            df15, df1h = md.get_frames(symbol, cfg)
            if df15.empty or df1h.empty:
                logger.warning("no market data for %s, skipping", symbol)
                continue

            pos = positions.get(symbol)
            last_ts = str(df15.index[-1])
            prev_ts = journal.get_state(f"last_candle_ts:{symbol}")
            new_candle = last_ts != prev_ts
            if not new_candle and pos is None:
                continue

            position_side = pos["side"] if pos else None
            decision = strategy.decide(symbol, df15, df1h, position_side)

            if decision.action in ("long", "short") and pos is None and new_candle:
                if in_cooldown(journal, symbol):
                    logger.info("%s in cooldown, skipping entry", symbol)
                else:
                    ok, why = risk.can_open(len(positions))
                    if not ok:
                        logger.info("cannot open %s: %s", symbol, why)
                    else:
                        instrument = trader.instrument(symbol)
                        if instrument is None:
                            logger.warning(
                                "no instrument filters for %s, skipping", symbol
                            )
                        else:
                            order = risk.calc_order(
                                decision.price, decision.atr,
                                "Buy" if decision.action == "long" else "Sell",
                                instrument,
                            )
                            if order is None:
                                logger.info("calc_order None for %s", symbol)
                            else:
                                side = (
                                    "Buy" if decision.action == "long" else "Sell"
                                )
                                trader.open_position(
                                    symbol, side, order["qty"],
                                    order["sl"], order["tp"],
                                )
                                reason = format_votes(decision.votes)
                                journal.log_open(
                                    symbol, side, order["qty"], decision.price,
                                    order["sl"], order["tp"], reason,
                                )
                                positions[symbol] = {
                                    "side": side, "qty": order["qty"],
                                    "entry": decision.price, "unrealized": 0.0,
                                }
                                side_ru = (
                                    "лонг 📈" if side == "Buy" else "шорт 📉"
                                )
                                tg_send(
                                    cfg,
                                    f"✅ {symbol}: открыт {side_ru}\n"
                                    f"Цена: {decision.price}\n"
                                    f"Объём: {order['qty']}\n"
                                    f"SL: {order['sl']} | TP: {order['tp']}\n"
                                    f"Скор: {decision.score:+.1f}\n\n{reason}",
                                )

            elif decision.action == "close" and pos is not None:
                close_since = int(time.time() * 1000) - 10_000
                if trader.close_position(symbol):
                    pnl = trader.last_closed_pnl(symbol, close_since)
                    journal.log_close(symbol, None, pnl, "сигнал развернулся")
                    set_cooldown(cfg, journal, symbol)
                    positions.pop(symbol, None)
                    pnl_txt = (
                        f"{'+' if pnl >= 0 else ''}{pnl:.2f} USDT"
                        if pnl is not None else "н/д"
                    )
                    tg_send(
                        cfg,
                        f"🔄 {symbol}: сигнал развернулся, позиция закрыта, "
                        f"PnL {pnl_txt}\n\n{format_votes(decision.votes)}",
                    )
                else:
                    logger.warning(
                        "close signal for %s but no position, deferring", symbol
                    )
                    continue  # reconcile will pick it up next run

            # Mark the candle consumed only after actions completed.
            journal.set_state(f"last_candle_ts:{symbol}", last_ts)

        except Exception:
            logger.exception("processing %s failed, continuing", symbol)

    # --- equity snapshot (every run; runs are ~15 min apart)
    equity = trader.get_equity()
    if equity > 0:
        journal.snapshot_equity(equity)


# ----------------------------------------------------------------------- main

def build_exchange(cfg, journal):
    """Return (market_data, trader, human-readable exchange name)."""
    if cfg.exchange == "paper":
        from exchange_paper import PaperMarketData, PaperTrader, create_session

        ex = create_session(cfg)
        return (
            PaperMarketData(ex),
            PaperTrader(ex, cfg, journal),
            "бумажная торговля (реальные цены, виртуальный счёт)",
        )

    if cfg.exchange == "okx":
        from exchange_okx import OkxMarketData, OkxTrader, create_session

        ex = create_session(cfg)
        return OkxMarketData(ex), OkxTrader(ex, cfg), "OKX Demo Trading"

    if cfg.exchange == "binance":
        from exchange_binance import (
            BinanceMarketData, BinanceTrader, create_session,
        )
        ex = create_session(cfg)
        return BinanceMarketData(ex), BinanceTrader(ex, cfg), "Binance Futures Testnet"

    from pybit.unified_trading import HTTP

    session = HTTP(
        demo=True, api_key=cfg.bybit_api_key, api_secret=cfg.bybit_api_secret
    )
    return MarketData(session), Trader(session, cfg), "Bybit DEMO"


def main() -> int:
    setup_logging()
    cfg = Config()

    if cfg.exchange == "paper":
        required = ()   # no exchange account at all
    elif cfg.exchange == "okx":
        required = (
            ("OKX_API_KEY", cfg.okx_api_key),
            ("OKX_API_SECRET", cfg.okx_api_secret),
            ("OKX_PASSPHRASE", cfg.okx_passphrase),
        )
    elif cfg.exchange == "binance":
        required = (
            ("BINANCE_API_KEY", cfg.binance_api_key),
            ("BINANCE_API_SECRET", cfg.binance_api_secret),
        )
    else:
        required = (
            ("BYBIT_API_KEY", cfg.bybit_api_key),
            ("BYBIT_API_SECRET", cfg.bybit_api_secret),
        )
    missing = [
        name for name, val in required + (
            ("TELEGRAM_TOKEN", cfg.telegram_token),
            ("TELEGRAM_CHAT_ID", cfg.telegram_chat_id),
        ) if not val
    ]
    if missing:
        print("Не заполнены переменные окружения: " + ", ".join(missing))
        return 1

    journal = Journal(cfg.db_path)
    md, trader, exchange_name = build_exchange(cfg, journal)
    strategy = Strategy(cfg)
    risk = RiskManager(cfg, journal)

    started = datetime.now(timezone.utc).strftime("%H:%M")
    logger.info("single pass started at %s UTC", started)

    # Greet the owner once (re-greet after an exchange switch).
    if journal.get_state("greeted") != f"1:{cfg.exchange}":
        llm_txt = "вкл" if (cfg.llm_enabled and cfg.anthropic_api_key) else "выкл"
        tg_send(
            cfg,
            f"🤖 Бот запущен в облаке (GitHub Actions, {exchange_name})\n"
            f"Пары: {', '.join(cfg.symbols)}\n"
            f"Плечо: x{cfg.leverage} | Бюджет: {cfg.budget_usdt:.0f} USDT\n"
            f"Цикл: раз в ~15 минут | LLM: {llm_txt}\n\n" + HELP_TEXT,
        )
        journal.set_state("greeted", f"1:{cfg.exchange}")

    try:
        trader.setup()
        process_commands(cfg, journal, trader)
        trading_pass(cfg, journal, md, strategy, risk, trader)
    except Exception as exc:
        logger.exception("run failed")
        # Throttle error notifications to at most one per hour.
        last = float(journal.get_state("last_error_notify", "0") or 0)
        if time.time() - last > 3600:
            journal.set_state("last_error_notify", str(time.time()))
            tg_send(
                cfg,
                f"⚠️ Ошибка в запуске бота: {exc}\n"
                "Следующая попытка через ~15 минут.",
            )
        return 0  # state must still be committed; don't fail the workflow

    logger.info("single pass finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
