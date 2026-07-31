"""Telegram control interface for the trading bot (aiogram 3.x).

Builds a Bot + Dispatcher wired to the shared deps. All user-facing strings
are in Russian; all blocking trader/journal calls run via asyncio.to_thread.
"""

import asyncio
import html
import logging
import time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

logger = logging.getLogger("bot.telegram")

HELP_TEXT = (
    "🤖 <b>Бот для торговли на Bybit (демо)</b>\n\n"
    "Доступные команды:\n"
    "/status — состояние бота, позиции, PnL за день\n"
    "/balance — баланс демо-счёта и бюджет бота\n"
    "/positions — открытые позиции подробно\n"
    "/pause — приостановить торговлю (позиции остаются)\n"
    "/resume — возобновить торговлю\n"
    "/close BTCUSDT — закрыть позицию по символу\n"
    "/close all — закрыть все позиции\n"
    "/report — отчёт по последним сделкам\n"
    "/help — этот список команд"
)


def _fmt_positions(positions: dict) -> str:
    """Format positions dict from Trader.get_positions() into Russian text."""
    if not positions:
        return "Открытых позиций нет."
    lines = []
    for symbol, p in positions.items():
        side_ru = "лонг 📈" if p.get("side") == "Buy" else "шорт 📉"
        upnl = float(p.get("unrealized", 0.0))
        sign = "+" if upnl >= 0 else ""
        lines.append(
            f"<b>{html.escape(str(symbol))}</b>: {side_ru}, "
            f"объём {p.get('qty')}, вход {p.get('entry')}, "
            f"uPnL {sign}{upnl:.2f} USDT"
        )
    return "\n".join(lines)


async def _safe_reply(message: Message, text: str) -> None:
    """Reply and swallow any Telegram error so handlers never crash."""
    try:
        await message.answer(text)
    except Exception:
        logger.exception("failed to send Telegram reply")


def build_bot(cfg, deps: dict) -> tuple[Bot, Dispatcher]:
    """Build Bot and Dispatcher with all command handlers.

    deps: {"state": BotState, "trader": Trader, "journal": Journal, "cfg": Config}
    """
    from aiogram.client.default import DefaultBotProperties

    bot = Bot(
        token=cfg.telegram_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    dp = Dispatcher()
    router = Router(name="commands")

    state = deps["state"]
    trader = deps["trader"]
    journal = deps["journal"]

    # Silent drop: only react to messages from the configured chat id.
    router.message.filter(F.chat.id == cfg.telegram_chat_id)

    @router.message(Command("start", "help"))
    async def cmd_help(message: Message) -> None:
        await _safe_reply(message, HELP_TEXT)

    @router.message(Command("status"))
    async def cmd_status(message: Message) -> None:
        try:
            positions = await asyncio.to_thread(trader.get_positions)
            daily_pnl = await asyncio.to_thread(journal.daily_realized_pnl)
            equity = await asyncio.to_thread(trader.get_equity)
            if state.paused:
                head = "⏸ Бот на паузе"
                if state.pause_reason:
                    head += f" ({html.escape(state.pause_reason)})"
            else:
                head = "▶️ Бот работает"
            sign = "+" if daily_pnl >= 0 else ""
            text = (
                f"{head}\n\n"
                f"{_fmt_positions(positions)}\n\n"
                f"PnL за сегодня: {sign}{daily_pnl:.2f} USDT\n"
                f"Эквити: {equity:.2f} USDT"
            )
            await _safe_reply(message, text)
        except Exception as e:
            logger.exception("/status failed")
            await _safe_reply(
                message, f"⚠️ Не удалось получить статус: {html.escape(str(e))}"
            )

    @router.message(Command("balance"))
    async def cmd_balance(message: Message) -> None:
        try:
            equity = await asyncio.to_thread(trader.get_equity)
            text = (
                f"💰 Эквити демо-счёта: {equity:.2f} USDT\n"
                f"Бюджет бота: {cfg.budget_usdt:.2f} USDT "
                f"(плечо x{cfg.leverage})"
            )
            await _safe_reply(message, text)
        except Exception as e:
            logger.exception("/balance failed")
            await _safe_reply(
                message, f"⚠️ Не удалось получить баланс: {html.escape(str(e))}"
            )

    @router.message(Command("positions"))
    async def cmd_positions(message: Message) -> None:
        try:
            positions = await asyncio.to_thread(trader.get_positions)
            await _safe_reply(message, "📊 " + _fmt_positions(positions))
        except Exception as e:
            logger.exception("/positions failed")
            await _safe_reply(
                message, f"⚠️ Не удалось получить позиции: {html.escape(str(e))}"
            )

    @router.message(Command("pause"))
    async def cmd_pause(message: Message) -> None:
        try:
            state.paused = True
            state.pause_reason = "пауза по команде пользователя"
            await _safe_reply(
                message,
                "⏸ Торговля приостановлена. Открытые позиции остаются, "
                "новые сделки не открываются. /resume — продолжить.",
            )
        except Exception as e:
            logger.exception("/pause failed")
            await _safe_reply(message, f"⚠️ Ошибка: {html.escape(str(e))}")

    @router.message(Command("resume"))
    async def cmd_resume(message: Message) -> None:
        try:
            state.paused = False
            state.pause_reason = ""
            await _safe_reply(message, "▶️ Торговля возобновлена.")
        except Exception as e:
            logger.exception("/resume failed")
            await _safe_reply(message, f"⚠️ Ошибка: {html.escape(str(e))}")

    @router.message(Command("close"))
    async def cmd_close(message: Message, command: CommandObject) -> None:
        try:
            arg = (command.args or "").strip()
            if not arg:
                await _safe_reply(
                    message,
                    "Укажите символ: /close BTCUSDT — или /close all, "
                    "чтобы закрыть все позиции.",
                )
                return

            positions = await asyncio.to_thread(trader.get_positions)
            if not positions:
                await _safe_reply(message, "Открытых позиций нет.")
                return

            if arg.lower() == "all":
                targets = list(positions.keys())
            else:
                symbol = arg.upper()
                if symbol not in positions:
                    await _safe_reply(
                        message,
                        f"По {html.escape(symbol)} нет открытой позиции.",
                    )
                    return
                targets = [symbol]

            closed, failed = [], []
            for symbol in targets:
                try:
                    # Margin absorbs local-vs-exchange clock skew; only a
                    # closed-pnl record newer than this is accepted.
                    close_since = int(time.time() * 1000) - 10_000
                    ok = await asyncio.to_thread(trader.close_position, symbol)
                    if not ok:
                        # No order was placed (position already gone or the
                        # lookup found nothing) — do not journal or report
                        # the position as closed by this command.
                        logger.warning(
                            "/close %s: no close order placed", symbol
                        )
                        failed.append(symbol)
                        continue
                    pnl = await asyncio.to_thread(
                        trader.last_closed_pnl, symbol, close_since
                    )
                    await asyncio.to_thread(
                        journal.log_close, symbol, None, pnl,
                        "закрыто вручную через Telegram",
                    )
                    if pnl is not None:
                        sign = "+" if pnl >= 0 else ""
                        closed.append(f"{symbol} (PnL {sign}{pnl:.2f} USDT)")
                    else:
                        closed.append(symbol)
                except Exception:
                    logger.exception("failed to close %s", symbol)
                    failed.append(symbol)

            parts = []
            if closed:
                parts.append("✅ Закрыто: " + ", ".join(closed))
            if failed:
                parts.append("⚠️ Не удалось закрыть: " + ", ".join(failed))
            await _safe_reply(message, "\n".join(parts) or "Нечего закрывать.")
        except Exception as e:
            logger.exception("/close failed")
            await _safe_reply(
                message, f"⚠️ Ошибка при закрытии: {html.escape(str(e))}"
            )

    @router.message(Command("report"))
    async def cmd_report(message: Message) -> None:
        try:
            text = await asyncio.to_thread(journal.report_text)
            await _safe_reply(message, text)
        except Exception as e:
            logger.exception("/report failed")
            await _safe_reply(
                message, f"⚠️ Не удалось построить отчёт: {html.escape(str(e))}"
            )

    dp.include_router(router)
    return bot, dp


async def notify(bot, cfg, text: str) -> None:
    """Send a message to the configured chat. Never raises."""
    try:
        await bot.send_message(cfg.telegram_chat_id, text)
    except Exception:
        logger.exception("notify failed")
