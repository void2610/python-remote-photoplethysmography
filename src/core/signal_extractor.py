"""
RGB信号抽出モジュール

ROI領域から時系列RGB信号を抽出し、バッファ管理を行います。
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional


class SignalExtractor:
    """
    ROIからRGB信号を抽出し、時系列データを管理するクラス
    """

    def __init__(self, buffer_size: int = 300):
        """
        コンストラクタ

        Args:
            buffer_size: 信号バッファサイズ（フレーム数）
                        例: 30 FPS × 10秒 = 300フレーム
        """
        self.buffer_size = buffer_size

        # RGB信号バッファ（dequeを使用して自動的に古いデータを削除）
        self.r_buffer = deque(maxlen=buffer_size)
        self.g_buffer = deque(maxlen=buffer_size)
        self.b_buffer = deque(maxlen=buffer_size)

        # タイムスタンプバッファ
        self.timestamp_buffer = deque(maxlen=buffer_size)

    def extract_rgb_signals(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        ROIからRGB信号の平均値を抽出

        Args:
            frame: BGR画像（OpenCV形式）
            roi_mask: ROIマスク（バイナリ）

        Returns:
            (r_mean, g_mean, b_mean): RGB平均値のタプル
            ROIに有効なピクセルがない場合はNone
        """
        # ROI領域のピクセルのみを抽出
        roi_pixels = frame[roi_mask > 0]

        # ROIが空の場合
        if len(roi_pixels) == 0:
            return None

        # BGR → RGB変換して平均値を計算
        b_mean = float(np.mean(roi_pixels[:, 0]))
        g_mean = float(np.mean(roi_pixels[:, 1]))
        r_mean = float(np.mean(roi_pixels[:, 2]))

        return r_mean, g_mean, b_mean

    def add_signal(
        self,
        r: float,
        g: float,
        b: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        RGB信号をバッファに追加

        Args:
            r: R（赤）チャンネルの平均値
            g: G（緑）チャンネルの平均値
            b: B（青）チャンネルの平均値
            timestamp: タイムスタンプ（秒単位、オプション）
        """
        self.r_buffer.append(r)
        self.g_buffer.append(g)
        self.b_buffer.append(b)

        if timestamp is not None:
            self.timestamp_buffer.append(timestamp)

    def get_signals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        バッファから信号を取得

        Returns:
            (r_signal, g_signal, b_signal): RGB信号の配列
        """
        r_signal = np.array(self.r_buffer)
        g_signal = np.array(self.g_buffer)
        b_signal = np.array(self.b_buffer)

        return r_signal, g_signal, b_signal

    def get_normalized_signals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        正規化されたRGB信号を取得（平均除算による正規化）

        Returns:
            (r_norm, g_norm, b_norm): 正規化されたRGB信号
        """
        r_signal, g_signal, b_signal = self.get_signals()

        # ゼロ除算を避けるための処理
        r_mean = np.mean(r_signal) if len(r_signal) > 0 else 1.0
        g_mean = np.mean(g_signal) if len(g_signal) > 0 else 1.0
        b_mean = np.mean(b_signal) if len(b_signal) > 0 else 1.0

        r_norm = r_signal / r_mean if r_mean != 0 else r_signal
        g_norm = g_signal / g_mean if g_mean != 0 else g_signal
        b_norm = b_signal / b_mean if b_mean != 0 else b_signal

        return r_norm, g_norm, b_norm

    def get_buffer_length(self) -> int:
        """
        現在のバッファ長を取得

        Returns:
            buffer_length: 現在のバッファに格納されているフレーム数
        """
        return len(self.r_buffer)

    def is_buffer_full(self) -> bool:
        """
        バッファが満杯かどうかを確認

        Returns:
            True: バッファが満杯
            False: バッファにまだ空きがある
        """
        return len(self.r_buffer) >= self.buffer_size

    def clear_buffer(self) -> None:
        """
        バッファをクリア
        """
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self.timestamp_buffer.clear()

    def get_timestamps(self) -> np.ndarray:
        """
        タイムスタンプの配列を取得

        Returns:
            timestamps: タイムスタンプの配列
        """
        return np.array(self.timestamp_buffer)


def extract_rgb_from_roi(
    frame: np.ndarray,
    roi_mask: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """
    ROIからRGB信号を抽出するユーティリティ関数

    Args:
        frame: BGR画像（OpenCV形式）
        roi_mask: ROIマスク

    Returns:
        (r_mean, g_mean, b_mean): RGB平均値
    """
    extractor = SignalExtractor(buffer_size=1)
    return extractor.extract_rgb_signals(frame, roi_mask)
