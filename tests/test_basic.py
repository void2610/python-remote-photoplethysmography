"""
基本的なユニットテスト
"""

import numpy as np
import pytest
from src.core.face_detector import FaceDetector
from src.core.signal_extractor import SignalExtractor
from src.algorithms.green import GreenAlgorithm
from src.signal_processing.filters import bandpass_filter, normalize_signal
from src.signal_processing.heart_rate_estimator import estimate_heart_rate_fft


class TestSignalExtractor:
    """SignalExtractorクラスのテスト"""

    def test_buffer_management(self):
        """バッファ管理のテスト"""
        extractor = SignalExtractor(buffer_size=10)

        # データを追加
        for i in range(5):
            extractor.add_signal(float(i), float(i), float(i))

        assert extractor.get_buffer_length() == 5
        assert not extractor.is_buffer_full()

        # さらに追加してバッファを満杯に
        for i in range(5, 15):
            extractor.add_signal(float(i), float(i), float(i))

        assert extractor.get_buffer_length() == 10
        assert extractor.is_buffer_full()

    def test_signal_retrieval(self):
        """信号取得のテスト"""
        extractor = SignalExtractor(buffer_size=5)

        # テストデータ
        test_data = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]

        for r, g, b in test_data:
            extractor.add_signal(r, g, b)

        r_signal, g_signal, b_signal = extractor.get_signals()

        assert len(r_signal) == 3
        assert len(g_signal) == 3
        assert len(b_signal) == 3
        assert r_signal[0] == 1.0
        assert g_signal[1] == 5.0
        assert b_signal[2] == 9.0


class TestGreenAlgorithm:
    """GreenAlgorithmクラスのテスト"""

    def test_process(self):
        """GREEN法の処理テスト"""
        algorithm = GreenAlgorithm()

        # テスト信号（緑チャンネルに心拍数72 BPMのサイン波）
        t = np.linspace(0, 10, 300)  # 10秒、30 FPS
        hr_true = 72  # BPM
        freq_true = hr_true / 60  # Hz

        r_signal = np.random.normal(128, 5, len(t))
        g_signal = 128 + 10 * np.sin(2 * np.pi * freq_true * t)
        b_signal = np.random.normal(128, 5, len(t))

        rgb_signals = (r_signal, g_signal, b_signal)

        # BVP信号を抽出
        bvp_signal = algorithm.process(rgb_signals)

        # 結果の検証
        assert len(bvp_signal) == len(g_signal)
        assert np.abs(np.mean(bvp_signal)) < 1e-10  # 平均が0に近い
        assert np.abs(np.std(bvp_signal) - 1.0) < 1e-10  # 標準偏差が1に近い


class TestFilters:
    """フィルタ関数のテスト"""

    def test_bandpass_filter(self):
        """バンドパスフィルタのテスト"""
        # テスト信号（1 Hz + 2 Hz + 5 Hz）
        fs = 30.0
        t = np.linspace(0, 10, int(10 * fs))
        signal = np.sin(2 * np.pi * 1.0 * t) + \
                 np.sin(2 * np.pi * 2.0 * t) + \
                 np.sin(2 * np.pi * 5.0 * t)

        # バンドパスフィルタ適用（0.75-3.5 Hz）
        filtered = bandpass_filter(signal, lowcut=0.75, highcut=3.5, fs=fs)

        # フィルタ後の信号長が元の信号と同じ
        assert len(filtered) == len(signal)

    def test_normalize_signal(self):
        """正規化のテスト"""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # 標準化
        normalized = normalize_signal(signal, method='standard')
        assert np.abs(np.mean(normalized)) < 1e-10
        assert np.abs(np.std(normalized) - 1.0) < 1e-10

        # 最小-最大正規化
        normalized = normalize_signal(signal, method='minmax')
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0


class TestHeartRateEstimator:
    """心拍数推定のテスト"""

    def test_estimate_heart_rate_fft(self):
        """FFT心拍数推定のテスト"""
        # 既知の心拍数でテスト信号を生成（72 BPM = 1.2 Hz）
        fs = 30.0
        hr_true = 72.0
        freq_true = hr_true / 60.0

        t = np.linspace(0, 10, int(10 * fs))
        bvp_signal = np.sin(2 * np.pi * freq_true * t)

        # 心拍数推定
        hr_estimated, confidence = estimate_heart_rate_fft(bvp_signal, fs=fs)

        # 誤差を確認（±5 BPM以内）
        assert np.abs(hr_estimated - hr_true) < 5.0
        assert confidence > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
