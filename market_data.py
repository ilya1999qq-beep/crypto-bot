"""Market data fetching and indicator computation for the Bybit demo bot.

All indicators are hand-rolled with pandas (no external TA libraries).
Wilder smoothing (RSI/ATR/ADX) is implemented via ewm(alpha=1/n, adjust=False).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("bot.market_data")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class MarketData:
    def __init__(self, session):
        # pybit.unified_trading.HTTP session (synchronous client)
        self.session = session

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        """Fetch closed candles for a symbol.

        Returns a DataFrame with float columns open/high/low/close/volume and a
        UTC DatetimeIndex. The last (still open) candle is dropped. On any
        error an empty DataFrame with the same columns is returned.
        """
        try:
            resp = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            rows = resp.get("result", {}).get("list", []) or []
        except Exception:
            logger.exception("get_kline failed for %s interval=%s", symbol, interval)
            return self._empty_frame()

        if not rows:
            logger.warning("get_kline returned no data for %s interval=%s", symbol, interval)
            return self._empty_frame()

        try:
            # Bybit returns newest-first: [startTime, open, high, low, close, volume, turnover]
            rows = list(reversed(rows))
            df = pd.DataFrame(
                [r[:6] for r in rows],
                columns=["start", "open", "high", "low", "close", "volume"],
            )
            df.index = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
            df.index.name = "time"
            df = df.drop(columns=["start"]).astype(float)
            # Drop the last (still open) candle.
            df = df.iloc[:-1]
            return df
        except Exception:
            logger.exception("failed to parse klines for %s interval=%s", symbol, interval)
            return self._empty_frame()

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC"))

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add standard indicator columns (pure pandas, no TA libraries).

        Columns: ema20, ema50, ema200, rsi14, macd, macd_signal, macd_hist,
        atr14, bb_upper, bb_mid, bb_lower, adx14, obv.
        """
        df = df.copy()
        indicator_cols = [
            "ema20", "ema50", "ema200", "rsi14",
            "macd", "macd_signal", "macd_hist", "atr14",
            "bb_upper", "bb_mid", "bb_lower", "adx14", "obv",
        ]
        if df.empty:
            for col in indicator_cols:
                df[col] = pd.Series(dtype=float)
            return df

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- EMA ---
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()
        df["ema200"] = close.ewm(span=200, adjust=False).mean()

        # --- RSI 14 (Wilder smoothing) ---
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
        # When avg_loss == 0: RSI is 100 if there were gains, 50 if flat.
        rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
        df["rsi14"] = rsi

        # --- MACD 12/26/9 ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd - macd_signal

        # --- True Range and ATR 14 (Wilder) ---
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        df["atr14"] = atr

        # --- Bollinger Bands 20/2 ---
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std(ddof=0)
        df["bb_mid"] = bb_mid
        df["bb_upper"] = bb_mid + 2.0 * bb_std
        df["bb_lower"] = bb_mid - 2.0 * bb_std

        # --- ADX 14 (Wilder) ---
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )
        plus_dm_smooth = plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        # Guard against zero ATR (flat market) to avoid division by zero.
        atr_safe = atr.replace(0.0, np.nan)
        plus_di = 100.0 * plus_dm_smooth / atr_safe
        minus_di = 100.0 * minus_dm_smooth / atr_safe
        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        df["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

        # --- OBV ---
        direction = np.sign(close.diff()).fillna(0.0)
        df["obv"] = (direction * volume).cumsum()

        return df

    def get_frames(self, symbol: str, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (df15, df1h) with indicators, per cfg.timeframe / cfg.trend_timeframe."""
        df15 = self.add_indicators(self.get_klines(symbol, cfg.timeframe))
        df1h = self.add_indicators(self.get_klines(symbol, cfg.trend_timeframe))
        return df15, df1h
