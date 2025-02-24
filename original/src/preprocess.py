import numpy as np


def calculate_rsi(prices, window=14):
    delta = np.diff(prices, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Smooth averages using EMA
    avg_gain = np.full_like(prices, np.nan)
    avg_loss = np.full_like(prices, np.nan)
    avg_gain[window] = np.mean(gain[1 : window + 1])
    avg_loss[window] = np.mean(loss[1 : window + 1])

    for i in range(window + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line for a 1D array of closing prices."""
    ema_fast = np.full_like(prices, np.nan)
    ema_slow = np.full_like(prices, np.nan)

    # Calculate EMAs
    ema_fast[fast - 1] = np.mean(prices[:fast])
    ema_slow[slow - 1] = np.mean(prices[:slow])

    for i in range(fast, len(prices)):
        ema_fast[i] = (prices[i] * (2 / (fast + 1))) + (
            ema_fast[i - 1] * (1 - 2 / (fast + 1))
        )
    for i in range(slow, len(prices)):
        ema_slow[i] = (prices[i] * (2 / (slow + 1))) + (
            ema_slow[i - 1] * (1 - 2 / (slow + 1))
        )

    macd = ema_fast - ema_slow
    signal_line = np.full_like(macd, np.nan)
    signal_line[slow + signal - 2] = np.mean(macd[slow - 1 : slow + signal - 1])

    for i in range(slow + signal - 1, len(prices)):
        signal_line[i] = (macd[i] * (2 / (signal + 1))) + (
            signal_line[i - 1] * (1 - 2 / (signal + 1))
        )
    return np.stack([macd, signal_line], axis=1)


def append_technical_indicators(stock_prices):
    close_prices = stock_prices[:, :, 3]
    rsi = np.apply_along_axis(calculate_rsi, axis=1, arr=close_prices)
    rsi = np.expand_dims(rsi, axis=-1)
    rsi[np.isnan(rsi)] = 50
    mca = np.apply_along_axis(
        calculate_macd,
        axis=1,
        arr=close_prices,
    )

    mca[np.isnan(mca)] = 0

    return np.concatenate([stock_prices, rsi, mca], axis=2)
