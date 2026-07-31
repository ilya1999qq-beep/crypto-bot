"""Entry point: wires together config, market data, strategy, risk, trader,
journal and the Telegram bot, then runs the polling loop and the trade loop.

No network calls and no object construction with side effects at import time —
everything is built inside main().
"""

import asyncio
import html
import logging
import sys
import time

from config import Config
from state import BotState
from market_data import MarketData
from strategy import Strategy, format_votes
from risk import RiskManager
from trader import Trader
from journal import Journal
from telegram_bot import build_bot, notify

logger = logging.getLogger("bot.main")

EQUITY_SNAPSHOT_SECONDS = 15 * 60   # snapshot equity every ~15 minutes
ERROR_NOTIFY_SECONDS = 10 * 60      # notify about loop errors at most once per 10 min


def setup_logging(cfg) -> None:
    """Configure console + file handlers on the root 'bot' logger."""
    root = logging.getLogger("bot")
    root.setLevel(logging.INFO)
    if root.handlers:  # avoid duplicate handlers on re-entry
        return
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        file_handler = logging.FileHandler(cfg.log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        root.exception("failed to open log file %s", cfg.log_path)


def validate_env(cfg) -> list[str]:
    """Return a list of missing required .env variables."""
    missing = []
    if not cfg.bybit_api_key:
        missing.append("BYBIT_API_KEY")
    if not cfg.bybit_api_secret:
        missing.append("BYBIT_API_SECRET")
    if not cfg.telegram_token:
        missing.append("TELEGRAM_TOKEN")
    if not cfg.telegram_chat_id:
        missing.append("TELEGRAM_CHAT_ID")
    return missing


def _cooldown_seconds(cfg) -> float:
    try:
        return cfg.cooldown_candles * int(cfg.timeframe) * 60
    except (TypeError, ValueError):
        return cfg.cooldown_candles * 15 * 60


async def trade_loop(cfg, state, md, strategy, risk, trader, journal, bot) -> None:
    """Main trading cycle, runs forever with cfg.poll_seconds between passes."""
    last_equity_snapshot = 0.0
    last_error_notify = 0.0
    cooldown_s = _cooldown_seconds(cfg)

    while True:
        try:
            if state.paused:
                await asyncio.sleep(cfg.poll_seconds)
                continue

            # Raises on API failure; the outer except then skips the whole
            # iteration (reconcile + entries) instead of acting on an
            # error-empty snapshot.
            positions = await asyncio.to_thread(trader.get_positions)

            # --- reconcile: journal says open, exchange says closed (SL/TP hit)
            for symbol in await asyncio.to_thread(journal.open_symbols):
                if symbol in positions:
                    continue
                # Only accept a closed-pnl record newer than the trade's open
                # time, so a previous trade's PnL is never journalled here.
                since_ms = await asyncio.to_thread(
                    journal.open_trade_since_ms, symbol
                )
                pnl = await asyncio.to_thread(
                    trader.last_closed_pnl, symbol, since_ms
                )
                await asyncio.to_thread(
                    journal.log_close, symbol, None, pnl, "SL/TP на бирже"
                )
                state.set_cooldown(symbol, cooldown_s)
                if pnl is not None:
                    sign = "+" if pnl >= 0 else ""
                    pnl_txt = f"{sign}{pnl:.2f} USDT"
                else:
                    pnl_txt = "н/д"
                await notify(
                    bot, cfg,
                    f"🎯 {symbol}: позиция закрыта на бирже по SL/TP, "
                    f"PnL {pnl_txt}",
                )
                logger.info("reconciled exchange-closed position %s pnl=%s", symbol, pnl)

            # --- daily loss guard (journal query + lock — keep off the loop)
            if await asyncio.to_thread(risk.daily_loss_breached) and not state.paused:
                state.paused = True
                state.pause_reason = "достигнут дневной лимит убытка"
                await notify(
                    bot, cfg,
                    "🛑 Достигнут дневной лимит убытка — торговля автоматически "
                    "приостановлена. /resume — возобновить вручную.",
                )
                logger.warning("daily loss limit breached, bot paused")
                continue

            # --- per-symbol analysis (each symbol isolated: one failing
            # symbol must not starve the others or lose its own signal)
            for symbol in cfg.symbols:
                prev_ts = state.last_candle_ts.get(symbol)
                try:
                    df15, df1h = await asyncio.to_thread(md.get_frames, symbol, cfg)
                    if df15.empty or df1h.empty:
                        logger.warning("no market data for %s, skipping", symbol)
                        continue

                    pos = positions.get(symbol)
                    last_ts = df15.index[-1]
                    new_candle = last_ts != state.last_candle_ts.get(symbol)
                    # No new closed candle and no open position -> nothing to do.
                    if not new_candle and pos is None:
                        continue
                    state.last_candle_ts[symbol] = last_ts

                    position_side = pos["side"] if pos else None
                    # Blocking work (pandas agents + possible sync LLM HTTP
                    # call) — must not run on the event loop.
                    decision = await asyncio.to_thread(
                        strategy.decide, symbol, df15, df1h, position_side
                    )

                    if decision.action in ("long", "short"):
                        if pos is not None:
                            continue
                        if not new_candle:
                            continue
                        if state.in_cooldown(symbol):
                            logger.info("%s in cooldown, skipping entry", symbol)
                            continue
                        ok, why = await asyncio.to_thread(
                            risk.can_open, len(positions)
                        )
                        if not ok:
                            logger.info("cannot open %s: %s", symbol, why)
                            continue

                        side = "Buy" if decision.action == "long" else "Sell"
                        instrument = await asyncio.to_thread(trader.instrument, symbol)
                        if instrument is None:
                            logger.warning(
                                "no instrument filters for %s, skipping entry", symbol
                            )
                            continue
                        order = risk.calc_order(
                            decision.price, decision.atr, side, instrument
                        )
                        if order is None:
                            logger.info("calc_order returned None for %s, skipping", symbol)
                            continue

                        await asyncio.to_thread(
                            trader.open_position, symbol, side,
                            order["qty"], order["sl"], order["tp"],
                        )
                        reason = format_votes(decision.votes)
                        await asyncio.to_thread(
                            journal.log_open, symbol, side, order["qty"],
                            decision.price, order["sl"], order["tp"], reason,
                        )
                        positions[symbol] = {
                            "side": side, "qty": order["qty"],
                            "entry": decision.price, "unrealized": 0.0,
                        }
                        side_ru = "лонг 📈" if side == "Buy" else "шорт 📉"
                        await notify(
                            bot, cfg,
                            f"✅ {symbol}: открыт {side_ru}\n"
                            f"Цена: {decision.price}\n"
                            f"Объём: {order['qty']}\n"
                            f"SL: {order['sl']} | TP: {order['tp']}\n"
                            f"Скор: {decision.score:+.1f}\n\n{reason}",
                        )

                    elif decision.action == "close" and pos is not None:
                        # Margin absorbs local-vs-exchange clock skew; only a
                        # closed-pnl record newer than this is accepted.
                        close_since = int(time.time() * 1000) - 10_000
                        closed = await asyncio.to_thread(
                            trader.close_position, symbol
                        )
                        if not closed:
                            # Position vanished (likely SL/TP just hit); the
                            # reconcile block will journal it next iteration.
                            logger.warning(
                                "close signal for %s but no position on "
                                "exchange, deferring to reconcile", symbol,
                            )
                            continue
                        pnl = await asyncio.to_thread(
                            trader.last_closed_pnl, symbol, close_since
                        )
                        await asyncio.to_thread(
                            journal.log_close, symbol, None, pnl, "сигнал развернулся"
                        )
                        state.set_cooldown(symbol, cooldown_s)
                        positions.pop(symbol, None)
                        if pnl is not None:
                            sign = "+" if pnl >= 0 else ""
                            pnl_txt = f"{sign}{pnl:.2f} USDT"
                        else:
                            pnl_txt = "н/д"
                        await notify(
                            bot, cfg,
                            f"🔄 {symbol}: сигнал развернулся, позиция закрыта, "
                            f"PnL {pnl_txt}\n\n{format_votes(decision.votes)}",
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Keep the candle "unconsumed" so the signal is retried on
                    # the next poll, and move on to the remaining symbols.
                    state.last_candle_ts[symbol] = prev_ts
                    logger.exception(
                        "processing %s failed, continuing with next symbol", symbol
                    )

            # --- periodic equity snapshot
            now = time.time()
            if now - last_equity_snapshot >= EQUITY_SNAPSHOT_SECONDS:
                equity = await asyncio.to_thread(trader.get_equity)
                if equity > 0:
                    await asyncio.to_thread(journal.snapshot_equity, equity)
                last_equity_snapshot = now

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("trade loop iteration failed")
            now = time.time()
            if now - last_error_notify >= ERROR_NOTIFY_SECONDS:
                last_error_notify = now
                await notify(
                    bot, cfg,
                    f"⚠️ Ошибка в торговом цикле: {html.escape(str(exc))}\n"
                    "Бот продолжает работу, подробности в логе.",
                )

        await asyncio.sleep(cfg.poll_seconds)


async def main() -> None:
    cfg = Config()
    setup_logging(cfg)

    missing = validate_env(cfg)
    if missing:
        print(
            "Ошибка: не заполнены обязательные переменные в .env:\n  - "
            + "\n  - ".join(missing)
            + "\nЗаполните их и запустите бота снова."
        )
        sys.exit(1)

    from pybit.unified_trading import HTTP

    session = HTTP(
        demo=True,
        api_key=cfg.bybit_api_key,
        api_secret=cfg.bybit_api_secret,
    )

    md = MarketData(session)
    strategy = Strategy(cfg)
    journal = Journal(cfg.db_path)
    risk = RiskManager(cfg, journal)
    trader = Trader(session, cfg)
    await asyncio.to_thread(trader.setup)
    state = BotState()

    deps = {"state": state, "trader": trader, "journal": journal, "cfg": cfg}
    bot, dp = build_bot(cfg, deps)

    llm_txt = "вкл" if (cfg.llm_enabled and cfg.anthropic_api_key) else "выкл"
    await notify(
        bot, cfg,
        "🤖 Бот запущен (Bybit DEMO)\n"
        f"Пары: {', '.join(cfg.symbols)}\n"
        f"Плечо: x{cfg.leverage} | Бюджет: {cfg.budget_usdt:.0f} USDT\n"
        f"Таймфрейм: {cfg.timeframe}м (тренд {cfg.trend_timeframe}м) | LLM: {llm_txt}",
    )
    logger.info("bot started: symbols=%s leverage=%s budget=%s",
                cfg.symbols, cfg.leverage, cfg.budget_usdt)

    await asyncio.gather(
        dp.start_polling(bot),
        trade_loop(cfg, state, md, strategy, risk, trader, journal, bot),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
