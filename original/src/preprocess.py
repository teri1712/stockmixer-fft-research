import numpy as np


def calculate_bollinger_bands(close_prices, window=20, num_std=2):
    sma = np.full_like(close_prices, np.nan)
    for i in range(window - 1, len(close_prices)):
        sma[i] = np.mean(close_prices[i - window + 1: i + 1])

    rolling_std = np.full_like(close_prices, np.nan)
    for i in range(window - 1, len(close_prices)):
        rolling_std[i] = np.std(close_prices[i - window + 1: i + 1])

    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    upper_band[np.isnan(upper_band)] = 0
    lower_band[np.isnan(lower_band)] = 0
    sma[np.isnan(sma)] = 0

    upper_band = (upper_band - upper_band.min()) / (upper_band.max() - upper_band.min())
    sma = (sma - sma.min()) / (sma.max() - sma.min())
    lower_band = (lower_band - lower_band.min()) / (lower_band.max() - lower_band.min())
    return np.stack([upper_band, sma, lower_band], axis=1)


def calculate_rsi(prices, window=14):
    delta = np.diff(prices, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.full_like(prices, np.nan)
    avg_loss = np.full_like(prices, np.nan)
    avg_gain[window] = np.mean(gain[1: window + 1])
    avg_loss[window] = np.mean(loss[1: window + 1])

    for i in range(window + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50

    return (rsi - rsi.min()) / (rsi.max() - rsi.min())


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = np.full_like(prices, np.nan)
    ema_slow = np.full_like(prices, np.nan)

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
    signal_line[slow + signal - 2] = np.mean(macd[slow - 1: slow + signal - 1])

    for i in range(slow + signal - 1, len(prices)):
        signal_line[i] = (macd[i] * (2 / (signal + 1))) + (
                signal_line[i - 1] * (1 - 2 / (signal + 1))
        )
    macd[np.isnan(macd)] = 0
    signal_line[np.isnan(signal_line)] = 0

    macd = (macd - macd.min()) / (macd.max() - macd.min())
    signal_line = (signal_line - signal_line.min()) / (
            signal_line.max() - signal_line.min()
    )

    return np.stack([macd], axis=1)


def append_technical_indicators(stock_prices):
    close_prices = stock_prices[:, :, 1]
    rsi = np.apply_along_axis(calculate_rsi, axis=1, arr=close_prices)
    rsi = np.expand_dims(rsi, axis=-1)
    # macd = np.apply_along_axis(
    #     calculate_macd,
    #     axis=1,
    #     arr=close_prices,
    # )
    # bb = np.apply_along_axis(
    #     calculate_bollinger_bands,
    #     axis=1,
    #     arr=close_prices,
    # )
    return np.concatenate([stock_prices, rsi], axis=2)
