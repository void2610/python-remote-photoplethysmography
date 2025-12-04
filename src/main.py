"""
リモート心拍数測定システムのメインアプリケーション

すべてのモジュールを統合し、リアルタイム心拍数測定を実現します。
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any

from src.core.face_detector import FaceDetector
from src.core.signal_extractor import SignalExtractor
from src.algorithms.green import GreenAlgorithm
from src.signal_processing.filters import bandpass_filter
from src.signal_processing.heart_rate_estimator import HeartRateEstimator


class HeartRateMonitor:
    """
    リモート心拍数モニタリングシステム

    Webカメラまたは動画ファイルから顔を検出し、
    rPPG技術を使って非接触で心拍数を測定します。
    """

    def __init__(
        self,
        algorithm: str = 'green',
        buffer_size: int = 300,
        fps: float = 30.0,
        estimation_method: str = 'fft'
    ):
        """
        コンストラクタ

        Args:
            algorithm: rPPGアルゴリズム ('green', 'chrom', 'pos', 'ica')
            buffer_size: 信号バッファサイズ（フレーム数）
            fps: カメラのFPS
            estimation_method: 心拍数推定手法 ('fft', 'peaks', 'autocorr', 'cwt')
        """
        # コンポーネントの初期化
        self.face_detector = FaceDetector()
        self.signal_extractor = SignalExtractor(buffer_size=buffer_size)
        self.hr_estimator = HeartRateEstimator(method=estimation_method, fs=fps)

        # アルゴリズムの選択
        if algorithm.lower() == 'green':
            self.rppg_algorithm = GreenAlgorithm()
        else:
            # 将来的に他のアルゴリズムを追加
            raise ValueError(f"Algorithm '{algorithm}' is not yet implemented")

        # パラメータ
        self.fps = fps
        self.buffer_size = buffer_size
        self.algorithm_name = algorithm

        # 状態管理
        self.is_running = False
        self.frame_count = 0
        self.last_hr = 0.0
        self.last_confidence = 0.0

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        単一フレームを処理

        Args:
            frame: BGR画像（OpenCV形式）

        Returns:
            result: 処理結果の辞書
                - heart_rate: 推定心拍数 (BPM)
                - confidence: 信頼度
                - bvp_signal: BVP信号
                - roi_mask: ROIマスク
                - landmarks: 顔のランドマーク
            顔が検出できなかった場合はNone
        """
        self.frame_count += 1

        # 1. 顔検出とROI抽出
        roi_mask, landmarks = self.face_detector.detect_roi(frame)

        if roi_mask is None:
            return None

        # 2. RGB信号抽出
        rgb_values = self.signal_extractor.extract_rgb_signals(frame, roi_mask)

        if rgb_values is None:
            return None

        r, g, b = rgb_values
        self.signal_extractor.add_signal(r, g, b)

        # バッファが十分溜まるまで待つ
        if not self.signal_extractor.is_buffer_full():
            return {
                'heart_rate': 0.0,
                'confidence': 0.0,
                'bvp_signal': np.array([]),
                'roi_mask': roi_mask,
                'landmarks': landmarks,
                'buffer_progress': self.signal_extractor.get_buffer_length() / self.buffer_size
            }

        # 3. 信号取得と正規化
        rgb_signals = self.signal_extractor.get_normalized_signals()

        # 4. rPPGアルゴリズム適用
        bvp_signal = self.rppg_algorithm.process(rgb_signals)

        # 5. バンドパスフィルタ
        bvp_filtered = bandpass_filter(bvp_signal, fs=self.fps)

        # 6. 心拍数推定
        hr, confidence = self.hr_estimator.estimate(bvp_filtered)

        # 結果を保存
        self.last_hr = hr
        self.last_confidence = confidence

        return {
            'heart_rate': hr,
            'confidence': confidence,
            'bvp_signal': bvp_filtered,
            'roi_mask': roi_mask,
            'landmarks': landmarks,
            'buffer_progress': 1.0
        }

    def visualize_result(
        self,
        frame: np.ndarray,
        result: Optional[Dict[str, Any]],
        show_roi: bool = True,
        show_landmarks: bool = False
    ) -> np.ndarray:
        """
        処理結果を画像に可視化

        Args:
            frame: 元のフレーム
            result: process_frameの結果
            show_roi: ROIを表示するか
            show_landmarks: ランドマークを表示するか

        Returns:
            vis_frame: 可視化されたフレーム
        """
        vis_frame = frame.copy()

        if result is None:
            # 顔が検出できなかった場合
            cv2.putText(
                vis_frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return vis_frame

        # ROI可視化
        if show_roi and result['roi_mask'] is not None:
            vis_frame = self.face_detector.visualize_roi(
                vis_frame,
                result['roi_mask'],
                result['landmarks'] if show_landmarks else None,
                color=(0, 255, 0),
                alpha=0.3
            )

        # 心拍数表示
        hr = result['heart_rate']
        confidence = result['confidence']
        buffer_progress = result.get('buffer_progress', 0.0)

        # バッファが溜まっていない場合
        if buffer_progress < 1.0:
            text = f"Buffering... {buffer_progress*100:.0f}%"
            color = (255, 165, 0)  # オレンジ
        else:
            # 信頼度に基づいて色を変更
            if confidence > 2.0:
                color = (0, 255, 0)  # 緑（高信頼度）
            elif confidence > 1.5:
                color = (255, 255, 0)  # 黄（中信頼度）
            else:
                color = (0, 165, 255)  # オレンジ（低信頼度）

            text = f"HR: {hr:.1f} BPM (Conf: {confidence:.2f})"

        cv2.putText(
            vis_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # アルゴリズム名表示
        cv2.putText(
            vis_frame,
            f"Algorithm: {self.algorithm_name.upper()}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # FPS表示
        cv2.putText(
            vis_frame,
            f"FPS: {self.fps:.1f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return vis_frame

    def run_webcam(
        self,
        camera_id: int = 0,
        show_visualization: bool = True,
        save_video: Optional[str] = None
    ) -> None:
        """
        Webカメラからリアルタイム測定

        Args:
            camera_id: カメラデバイスID
            show_visualization: 可視化を表示するか
            save_video: 動画保存パス（Noneの場合は保存しない）
        """
        # カメラ初期化
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 実際のFPSを取得
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            self.fps = actual_fps

        # 動画保存の設定
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            video_writer = cv2.VideoWriter(save_video, fourcc, self.fps, frame_size)

        self.is_running = True
        print(f"Starting heart rate monitoring (FPS: {self.fps:.1f})...")
        print("Press 'q' to quit, 'r' to reset")

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # フレーム処理
                result = self.process_frame(frame)

                # 可視化
                if show_visualization:
                    vis_frame = self.visualize_result(frame, result)
                    cv2.imshow('Heart Rate Monitor', vis_frame)

                    # 動画保存
                    if video_writer is not None:
                        video_writer.write(vis_frame)

                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # リセット
                    self.signal_extractor.clear_buffer()
                    self.hr_estimator.reset()
                    print("Buffer reset")

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            self.is_running = False
            print("Heart rate monitoring stopped")

    def run_video_file(
        self,
        video_path: str,
        show_visualization: bool = True,
        save_video: Optional[str] = None
    ) -> None:
        """
        動画ファイルから心拍数測定

        Args:
            video_path: 動画ファイルのパス
            show_visualization: 可視化を表示するか
            save_video: 動画保存パス（Noneの場合は保存しない）
        """
        # 動画ファイルを開く
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # FPSを取得
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        # 動画保存の設定
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            video_writer = cv2.VideoWriter(save_video, fourcc, self.fps, frame_size)

        self.is_running = True
        print(f"Processing video: {video_path} (FPS: {self.fps:.1f})")

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # フレーム処理
                result = self.process_frame(frame)

                # 可視化
                if show_visualization:
                    vis_frame = self.visualize_result(frame, result)
                    cv2.imshow('Heart Rate Monitor', vis_frame)

                    # 動画保存
                    if video_writer is not None:
                        video_writer.write(vis_frame)

                    # キー入力処理
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            self.is_running = False
            print("Video processing completed")

    def get_status(self) -> Dict[str, Any]:
        """
        現在の状態を取得

        Returns:
            status: 状態情報の辞書
        """
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'last_heart_rate': self.last_hr,
            'last_confidence': self.last_confidence,
            'buffer_length': self.signal_extractor.get_buffer_length(),
            'buffer_full': self.signal_extractor.is_buffer_full(),
            'algorithm': self.algorithm_name,
            'fps': self.fps
        }


if __name__ == "__main__":
    # テスト実行
    monitor = HeartRateMonitor(algorithm='green', buffer_size=300)
    monitor.run_webcam()
