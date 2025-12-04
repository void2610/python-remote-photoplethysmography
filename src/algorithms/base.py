"""
rPPGアルゴリズムの基底クラス

すべてのrPPGアルゴリズムが継承する抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class RPPGAlgorithm(ABC):
    """
    rPPG（リモート心拍測定）アルゴリズムの抽象基底クラス
    """

    def __init__(self, name: str):
        """
        コンストラクタ

        Args:
            name: アルゴリズム名
        """
        self.name = name

    @abstractmethod
    def process(
        self,
        rgb_signals: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        RGB信号からBVP（血液容積脈波）信号を抽出

        Args:
            rgb_signals: (r_signal, g_signal, b_signal)のタプル

        Returns:
            bvp_signal: 抽出されたBVP信号
        """
        pass

    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.name} Algorithm"

    def __repr__(self) -> str:
        """オブジェクト表現"""
        return f"RPPGAlgorithm(name='{self.name}')"
