"""
信号処理フィルタモジュール

BVP（血液容積脈波）信号のフィルタリング処理を行います。
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend
from typing import Optional


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.75,
    highcut: float = 3.5,
    fs: float = 30.0,
    order: int = 4
) -> np.ndarray:
    """
    バンドパスフィルタを適用

    生理学的に妥当な心拍数範囲（45-210 BPM = 0.75-3.5 Hz）の信号のみを通過させます。

    Args:
        signal: 入力信号
        lowcut: 下限周波数 (Hz)、デフォルト: 0.75 Hz (45 BPM)
        highcut: 上限周波数 (Hz)、デフォルト: 3.5 Hz (210 BPM)
        fs: サンプリング周波数 (Hz)、デフォルト: 30.0 Hz (30 FPS)
        order: フィルタ次数、デフォルト: 4

    Returns:
        filtered_signal: フィルタ後の信号
    """
    # 信号が短すぎる場合はそのまま返す
    if len(signal) < order * 3:
        return signal

    # デトレンディング（線形トレンド除去）
    signal_detrended = detrend(signal, type='linear')

    # Butterworthフィルタの設計（Second-Order Sections形式）
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

    # ゼロ位相フィルタリング（前後両方向にフィルタを適用）
    filtered_signal = sosfiltfilt(sos, signal_detrended)

    return filtered_signal


def normalize_signal(signal: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    信号を正規化

    Args:
        signal: 入力信号
        method: 正規化方法
            - 'standard': 標準化（平均0、標準偏差1）
            - 'minmax': 最小-最大正規化（0-1の範囲）
            - 'mean': 平均除算

    Returns:
        normalized_signal: 正規化された信号
    """
    if len(signal) == 0:
        return signal

    if method == 'standard':
        # 標準化（平均0、標準偏差1）
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean

    elif method == 'minmax':
        # 最小-最大正規化
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val > min_val:
            return (signal - min_val) / (max_val - min_val)
        else:
            return signal - min_val

    elif method == 'mean':
        # 平均除算
        mean = np.mean(signal)
        if mean > 0:
            return signal / mean
        else:
            return signal

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    移動平均フィルタを適用

    Args:
        signal: 入力信号
        window_size: 移動平均のウィンドウサイズ

    Returns:
        smoothed_signal: 平滑化された信号
    """
    if len(signal) < window_size:
        return signal

    # NumPyのconvolveを使用した移動平均
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, kernel, mode='same')

    return smoothed_signal


def apply_preprocessing(
    signal: np.ndarray,
    fs: float = 30.0,
    apply_bandpass: bool = True,
    apply_normalize: bool = True,
    normalize_method: str = 'standard'
) -> np.ndarray:
    """
    前処理パイプライン（デトレンディング→バンドパス→正規化）

    Args:
        signal: 入力信号
        fs: サンプリング周波数 (Hz)
        apply_bandpass: バンドパスフィルタを適用するか
        apply_normalize: 正規化を適用するか
        normalize_method: 正規化方法

    Returns:
        processed_signal: 前処理後の信号
    """
    processed_signal = signal.copy()

    # バンドパスフィルタ
    if apply_bandpass:
        processed_signal = bandpass_filter(processed_signal, fs=fs)

    # 正規化
    if apply_normalize:
        processed_signal = normalize_signal(processed_signal, method=normalize_method)

    return processed_signal


def detect_outliers(
    signal: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    外れ値を検出（標準偏差ベース）

    Args:
        signal: 入力信号
        threshold: 外れ値判定の閾値（標準偏差の倍数）

    Returns:
        outlier_mask: 外れ値のマスク（True: 外れ値、False: 正常値）
    """
    if len(signal) == 0:
        return np.array([], dtype=bool)

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return np.zeros(len(signal), dtype=bool)

    z_scores = np.abs((signal - mean) / std)
    outlier_mask = z_scores > threshold

    return outlier_mask


def remove_outliers(
    signal: np.ndarray,
    threshold: float = 3.0,
    method: str = 'interpolate'
) -> np.ndarray:
    """
    外れ値を除去または補間

    Args:
        signal: 入力信号
        threshold: 外れ値判定の閾値（標準偏差の倍数）
        method: 除去方法
            - 'interpolate': 線形補間で置き換え
            - 'median': 中央値で置き換え

    Returns:
        cleaned_signal: 外れ値が処理された信号
    """
    outlier_mask = detect_outliers(signal, threshold)
    cleaned_signal = signal.copy()

    if not np.any(outlier_mask):
        return cleaned_signal

    if method == 'interpolate':
        # 線形補間
        x = np.arange(len(signal))
        valid_idx = ~outlier_mask
        if np.sum(valid_idx) >= 2:  # 少なくとも2点必要
            cleaned_signal[outlier_mask] = np.interp(
                x[outlier_mask],
                x[valid_idx],
                signal[valid_idx]
            )

    elif method == 'median':
        # 中央値で置き換え
        median_val = np.median(signal[~outlier_mask])
        cleaned_signal[outlier_mask] = median_val

    return cleaned_signal
