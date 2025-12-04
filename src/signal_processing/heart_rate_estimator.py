"""
心拍数推定モジュール

BVP信号から心拍数（BPM）を推定します。
FFT、自己相関、ウェーブレット変換などの手法を提供します。
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional
import pywt


def estimate_heart_rate_fft(
    bvp_signal: np.ndarray,
    fs: float = 30.0,
    lowcut: float = 0.75,
    highcut: float = 3.5
) -> Tuple[float, float]:
    """
    FFTによる心拍数推定

    Args:
        bvp_signal: BVP信号
        fs: サンプリング周波数 (Hz)
        lowcut: 下限周波数 (Hz)、45 BPM に対応
        highcut: 上限周波数 (Hz)、210 BPM に対応

    Returns:
        heart_rate: 推定心拍数 (BPM)
        confidence: 信頼度スコア（ピークの突出度）
    """
    # 信号が短すぎる場合
    if len(bvp_signal) < 2:
        return 0.0, 0.0

    # FFT計算
    N = len(bvp_signal)
    fft_result = np.fft.rfft(bvp_signal)
    freqs = np.fft.rfftfreq(N, 1/fs)

    # 生理学的範囲でマスク（0.75-3.5 Hz = 45-210 BPM）
    mask = (freqs >= lowcut) & (freqs <= highcut)

    if not np.any(mask):
        return 0.0, 0.0

    freqs_masked = freqs[mask]
    fft_masked = np.abs(fft_result[mask])

    # パワースペクトルが空の場合
    if len(fft_masked) == 0:
        return 0.0, 0.0

    # ピーク周波数を検出
    peak_idx = np.argmax(fft_masked)
    peak_freq = freqs_masked[peak_idx]

    # BPM変換
    heart_rate = peak_freq * 60.0

    # 信頼度（ピークの突出度）
    mean_power = np.mean(fft_masked)
    if mean_power > 0:
        confidence = fft_masked[peak_idx] / mean_power
    else:
        confidence = 0.0

    return heart_rate, confidence


def estimate_heart_rate_peaks(
    bvp_signal: np.ndarray,
    fs: float = 30.0,
    min_distance: Optional[int] = None
) -> Tuple[float, float]:
    """
    ピーク検出による心拍数推定

    Args:
        bvp_signal: BVP信号
        fs: サンプリング周波数 (Hz)
        min_distance: ピーク間の最小距離（サンプル数）

    Returns:
        heart_rate: 推定心拍数 (BPM)
        confidence: 信頼度スコア
    """
    # デフォルトの最小距離（最大210 BPMに対応）
    if min_distance is None:
        min_distance = int(fs * 60 / 210)  # 210 BPM = 3.5 Hz

    # 信号が短すぎる場合
    if len(bvp_signal) < min_distance * 2:
        return 0.0, 0.0

    # ピーク検出
    peaks, properties = find_peaks(bvp_signal, distance=min_distance)

    # ピークが見つからない場合
    if len(peaks) < 2:
        return 0.0, 0.0

    # ピーク間隔から心拍数を推定
    peak_intervals = np.diff(peaks) / fs  # 秒単位
    avg_interval = np.mean(peak_intervals)

    if avg_interval > 0:
        heart_rate = 60.0 / avg_interval
    else:
        return 0.0, 0.0

    # 信頼度（ピーク間隔の一貫性）
    std_interval = np.std(peak_intervals)
    if avg_interval > 0:
        confidence = 1.0 / (1.0 + std_interval / avg_interval)
    else:
        confidence = 0.0

    return heart_rate, confidence


def estimate_heart_rate_autocorr(
    bvp_signal: np.ndarray,
    fs: float = 30.0
) -> float:
    """
    自己相関による心拍数推定

    Args:
        bvp_signal: BVP信号
        fs: サンプリング周波数 (Hz)

    Returns:
        heart_rate: 推定心拍数 (BPM)
    """
    # 信号が短すぎる場合
    if len(bvp_signal) < 2:
        return 0.0

    # 自己相関計算
    autocorr = np.correlate(bvp_signal, bvp_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # 正のラグのみ

    # 生理学的範囲のラグ（45-210 BPM）
    min_lag = int(fs * 60 / 210)  # 210 BPM に対応
    max_lag = int(fs * 60 / 45)   # 45 BPM に対応

    # ラグ範囲が有効かチェック
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    if min_lag >= max_lag:
        return 0.0

    # 範囲内でピーク検出
    autocorr_range = autocorr[min_lag:max_lag]

    if len(autocorr_range) == 0:
        return 0.0

    peak_lag = np.argmax(autocorr_range) + min_lag

    # BPM計算
    if peak_lag > 0:
        heart_rate = 60.0 * fs / peak_lag
    else:
        heart_rate = 0.0

    return heart_rate


def estimate_heart_rate_cwt(
    bvp_signal: np.ndarray,
    fs: float = 30.0,
    wavelet: str = 'morl'
) -> Tuple[float, np.ndarray]:
    """
    連続ウェーブレット変換による心拍数推定

    論文手法: Bousefsaf et al. (2013)

    Args:
        bvp_signal: BVP信号
        fs: サンプリング周波数 (Hz)
        wavelet: ウェーブレット種類

    Returns:
        heart_rate: 平均心拍数 (BPM)
        instantaneous_hr: 瞬時心拍数の時系列
    """
    # 信号が短すぎる場合
    if len(bvp_signal) < 2:
        return 0.0, np.array([])

    # スケール範囲設定（対応周波数: 0.75-3.5 Hz）
    # pywt.scale2frequencyを使ってスケールを周波数に変換
    scales = np.arange(1, 128)

    try:
        # CWT計算
        coefficients, frequencies = pywt.cwt(bvp_signal, scales, wavelet, 1/fs)

        # 周波数範囲でマスク
        freq_mask = (frequencies >= 0.75) & (frequencies <= 3.5)

        if not np.any(freq_mask):
            return 0.0, np.array([])

        # マスク適用
        coefficients_masked = coefficients[freq_mask, :]
        frequencies_masked = frequencies[freq_mask]

        # 各時刻の主要周波数を検出
        power = np.abs(coefficients_masked) ** 2
        dominant_freq_idx = np.argmax(power, axis=0)

        # 瞬時心拍数計算
        instantaneous_hr = frequencies_masked[dominant_freq_idx] * 60.0

        # 平均心拍数
        heart_rate = float(np.median(instantaneous_hr))

        return heart_rate, instantaneous_hr

    except Exception as e:
        # エラーが発生した場合
        return 0.0, np.array([])


class HeartRateEstimator:
    """
    心拍数推定器クラス

    複数の推定手法をサポートし、結果の平滑化や信頼度評価を行います。
    """

    def __init__(
        self,
        method: str = 'fft',
        fs: float = 30.0,
        smoothing_window: int = 3
    ):
        """
        コンストラクタ

        Args:
            method: 推定手法 ('fft', 'peaks', 'autocorr', 'cwt')
            fs: サンプリング周波数 (Hz)
            smoothing_window: 移動平均のウィンドウサイズ
        """
        self.method = method
        self.fs = fs
        self.smoothing_window = smoothing_window
        self.hr_history = []

    def estimate(self, bvp_signal: np.ndarray) -> Tuple[float, float]:
        """
        心拍数を推定

        Args:
            bvp_signal: BVP信号

        Returns:
            heart_rate: 推定心拍数 (BPM)
            confidence: 信頼度スコア
        """
        if self.method == 'fft':
            hr, conf = estimate_heart_rate_fft(bvp_signal, self.fs)
        elif self.method == 'peaks':
            hr, conf = estimate_heart_rate_peaks(bvp_signal, self.fs)
        elif self.method == 'autocorr':
            hr = estimate_heart_rate_autocorr(bvp_signal, self.fs)
            conf = 1.0
        elif self.method == 'cwt':
            hr, _ = estimate_heart_rate_cwt(bvp_signal, self.fs)
            conf = 1.0
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 履歴に追加
        self.hr_history.append(hr)

        # 平滑化
        if len(self.hr_history) > self.smoothing_window:
            self.hr_history = self.hr_history[-self.smoothing_window:]

        smoothed_hr = np.mean(self.hr_history)

        return smoothed_hr, conf

    def reset(self):
        """履歴をリセット"""
        self.hr_history = []
