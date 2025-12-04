"""
GREEN法アルゴリズム

最もシンプルなrPPG手法で、緑チャンネルのみを使用します。
緑光はヘモグロビンによる吸収が大きく、心拍による皮膚色の変化を最もよく捉えます。
"""

import numpy as np
from typing import Tuple
from .base import RPPGAlgorithm


class GreenAlgorithm(RPPGAlgorithm):
    """
    GREEN法の実装

    緑チャンネルの信号のみを使用してBVP信号を抽出します。
    実装が簡単で、基本的なベースライン手法として有用です。
    """

    def __init__(self):
        """コンストラクタ"""
        super().__init__(name="GREEN")

    def process(
        self,
        rgb_signals: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        RGB信号からBVP信号を抽出（GREEN法）

        Args:
            rgb_signals: (r_signal, g_signal, b_signal)のタプル

        Returns:
            bvp_signal: 抽出されたBVP信号（正規化済み）
        """
        _, g_signal, _ = rgb_signals

        # 信号が空の場合
        if len(g_signal) == 0:
            return np.array([])

        # 緑チャンネルを標準化（平均0、標準偏差1）
        mean = np.mean(g_signal)
        std = np.std(g_signal)

        if std > 0:
            bvp_signal = (g_signal - mean) / std
        else:
            bvp_signal = g_signal - mean

        return bvp_signal


def green_method(rgb_signals: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    GREEN法のユーティリティ関数

    Args:
        rgb_signals: (r_signal, g_signal, b_signal)のタプル

    Returns:
        bvp_signal: BVP信号
    """
    algorithm = GreenAlgorithm()
    return algorithm.process(rgb_signals)
