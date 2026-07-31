"""Trade journal backed by sqlite3.

Stores opened/closed trades and equity snapshots. The sqlite connection is
created with check_same_thread=False and every DB operation is guarded by a
threading.Lock, because the journal is called both from the async trade loop
(via asyncio.to_thread) and from Telegram handlers.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger("bot.journal")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_open      TEXT NOT NULL,
    ts_close     TEXT,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    qty          REAL NOT NULL,
    entry        REAL NOT NULL,
    exit         REAL,
    sl           REAL,
    tp           REAL,
    pnl          REAL,
    reason_open  TEXT,
    reason_close TEXT,
    status       TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS equity (
    ts    TEXT NOT NULL,
    value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _utcnow() -> str:
    """Current UTC time as ISO string (seconds precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Journal:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ trades

    def log_open(self, symbol, side, qty, entry, sl, tp, reason: str) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO trades (ts_open, symbol, side, qty, entry, sl, tp,"
                    " reason_open, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                    (_utcnow(), symbol, side, float(qty), float(entry),
                     float(sl), float(tp), reason),
                )
                self._conn.commit()
        except Exception:
            logger.exception("failed to log open trade for %s", symbol)

    def log_close(self, symbol, exit_price: float | None, pnl: float | None,
                  reason: str) -> None:
        """Close the latest open trade for the symbol."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT id FROM trades WHERE symbol = ? AND status = 'open'"
                    " ORDER BY id DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
                if row is None:
                    logger.warning("log_close: no open trade found for %s", symbol)
                    return
                self._conn.execute(
                    "UPDATE trades SET ts_close = ?, exit = ?, pnl = ?,"
                    " reason_close = ?, status = 'closed' WHERE id = ?",
                    (_utcnow(),
                     None if exit_price is None else float(exit_price),
                     None if pnl is None else float(pnl),
                     reason, row["id"]),
                )
                self._conn.commit()
        except Exception:
            logger.exception("failed to log close trade for %s", symbol)

    def has_open_trade(self, symbol: str) -> bool:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT 1 FROM trades WHERE symbol = ? AND status = 'open' LIMIT 1",
                    (symbol,),
                ).fetchone()
            return row is not None
        except Exception:
            logger.exception("has_open_trade failed for %s", symbol)
            return False

    def open_trade_since_ms(self, symbol: str) -> int | None:
        """Epoch ms of ts_open for the latest open trade on symbol, or None.

        Used to correlate Bybit closed-pnl records with the journalled trade:
        any record older than the trade's open time belongs to a previous trade.
        """
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT ts_open FROM trades WHERE symbol = ? AND status = 'open'"
                    " ORDER BY id DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
            if row is None or not row["ts_open"]:
                return None
            dt = datetime.strptime(
                row["ts_open"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            logger.exception("open_trade_since_ms failed for %s", symbol)
            return None

    def open_symbols(self) -> list[str]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT DISTINCT symbol FROM trades WHERE status = 'open'"
                ).fetchall()
            return [r["symbol"] for r in rows]
        except Exception:
            logger.exception("open_symbols failed")
            return []

    def daily_realized_pnl(self) -> float:
        """Sum of pnl for trades closed today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades"
                    " WHERE status = 'closed' AND pnl IS NOT NULL"
                    " AND ts_close LIKE ?",
                    (today + "%",),
                ).fetchone()
            return float(row["total"] or 0.0)
        except Exception:
            logger.exception("daily_realized_pnl failed")
            return 0.0

    # ------------------------------------------------------------------- state

    def get_state(self, key: str, default: str | None = None) -> str | None:
        """Persistent key-value state (used by the single-pass cron runner)."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT value FROM state WHERE key = ?", (key,)
                ).fetchone()
            return row["value"] if row is not None else default
        except Exception:
            logger.exception("get_state failed for %s", key)
            return default

    def set_state(self, key: str, value: str) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO state (key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )
                self._conn.commit()
        except Exception:
            logger.exception("set_state failed for %s", key)

    # ------------------------------------------------------------------ equity

    def snapshot_equity(self, equity: float) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO equity (ts, value) VALUES (?, ?)",
                    (_utcnow(), float(equity)),
                )
                self._conn.commit()
        except Exception:
            logger.exception("snapshot_equity failed")

    # ------------------------------------------------------------------ report

    def report_text(self) -> str:
        """Russian report: last 10 closed trades, totals, winrate."""
        try:
            with self._lock:
                closed = self._conn.execute(
                    "SELECT * FROM trades WHERE status = 'closed'"
                    " ORDER BY id DESC LIMIT 10"
                ).fetchall()
                stats = self._conn.execute(
                    "SELECT COUNT(*) AS n,"
                    " COALESCE(SUM(pnl), 0) AS total,"
                    " SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins"
                    " FROM trades WHERE status = 'closed' AND pnl IS NOT NULL"
                ).fetchone()
                open_count = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM trades WHERE status = 'open'"
                ).fetchone()["n"]
        except Exception:
            logger.exception("report_text failed")
            return "Ошибка при чтении журнала сделок."

        if not closed and open_count == 0:
            return "Журнал пуст — сделок пока не было."

        lines = ["📒 Отчёт по сделкам", ""]
        if closed:
            lines.append("Последние закрытые сделки:")
            for t in closed:
                side_ru = "лонг" if t["side"] == "Buy" else "шорт"
                pnl = t["pnl"]
                if pnl is None:
                    pnl_str = "PnL: н/д"
                else:
                    sign = "+" if pnl >= 0 else ""
                    pnl_str = f"PnL: {sign}{pnl:.2f} USDT"
                ts = t["ts_close"] or t["ts_open"] or ""
                reason = t["reason_close"] or ""
                line = f"• {ts} {t['symbol']} {side_ru} — {pnl_str}"
                if reason:
                    line += f" ({reason})"
                lines.append(line)
            lines.append("")

        n = stats["n"] or 0
        total = float(stats["total"] or 0.0)
        wins = stats["wins"] or 0
        if n > 0:
            winrate = wins / n * 100.0
            sign = "+" if total >= 0 else ""
            lines.append(f"Всего закрыто сделок: {n}")
            lines.append(f"Суммарный PnL: {sign}{total:.2f} USDT")
            lines.append(f"Прибыльных: {wins} из {n} (винрейт {winrate:.0f}%)")
        daily = self.daily_realized_pnl()
        d_sign = "+" if daily >= 0 else ""
        lines.append(f"Реализованный PnL за сегодня: {d_sign}{daily:.2f} USDT")
        if open_count:
            lines.append(f"Открытых сделок сейчас: {open_count}")
        return "\n".join(lines)
