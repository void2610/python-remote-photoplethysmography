"""
顔検出とROI抽出モジュール

MediaPipe Face Meshを使用して顔を検出し、
rPPG測定に適した領域（額、頬）を抽出します。
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


class FaceDetector:
    """
    MediaPipe Face Meshを使用した顔検出クラス

    468個のランドマークを検出し、額と頬のROIを抽出します。
    """

    # 額のランドマークインデックス
    FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356]

    # 左頬のランドマークインデックス
    LEFT_CHEEK_LANDMARKS = [50, 36, 206, 216, 212, 202]

    # 右頬のランドマークインデックス
    RIGHT_CHEEK_LANDMARKS = [280, 266, 426, 436, 432, 422]

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        コンストラクタ

        Args:
            max_num_faces: 検出する最大顔数
            min_detection_confidence: 検出の最小信頼度
            min_tracking_confidence: トラッキングの最小信頼度
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        フレームから顔のランドマークを検出

        Args:
            frame: BGR画像（OpenCV形式）

        Returns:
            landmarks: 検出されたランドマーク座標の配列 (468, 2)
                      検出できなかった場合はNone
        """
        # BGRからRGBに変換（MediaPipeはRGBを期待）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 顔検出を実行
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # 最初の顔のランドマークを取得
        face_landmarks = results.multi_face_landmarks[0]

        # ランドマークを画像座標に変換
        h, w = frame.shape[:2]
        landmarks = np.array([
            [landmark.x * w, landmark.y * h]
            for landmark in face_landmarks.landmark
        ])

        return landmarks

    def extract_roi_mask(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        use_forehead: bool = True,
        use_cheeks: bool = True
    ) -> Optional[np.ndarray]:
        """
        ランドマークからROIマスクを生成

        Args:
            frame: BGR画像（OpenCV形式）
            landmarks: 顔のランドマーク座標 (468, 2)
            use_forehead: 額領域を使用するか
            use_cheeks: 頬領域を使用するか

        Returns:
            roi_mask: バイナリマスク（0または255）
                     検出できなかった場合はNone
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 額のROI
        if use_forehead:
            forehead_points = landmarks[self.FOREHEAD_LANDMARKS].astype(np.int32)
            cv2.fillPoly(mask, [forehead_points], 255)

        # 左頬のROI
        if use_cheeks:
            left_cheek_points = landmarks[self.LEFT_CHEEK_LANDMARKS].astype(np.int32)
            cv2.fillPoly(mask, [left_cheek_points], 255)

            # 右頬のROI
            right_cheek_points = landmarks[self.RIGHT_CHEEK_LANDMARKS].astype(np.int32)
            cv2.fillPoly(mask, [right_cheek_points], 255)

        return mask

    def detect_roi(
        self,
        frame: np.ndarray,
        use_forehead: bool = True,
        use_cheeks: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        顔のROIを検出（ワンステップで実行）

        Args:
            frame: BGR画像（OpenCV形式）
            use_forehead: 額領域を使用するか
            use_cheeks: 頬領域を使用するか

        Returns:
            roi_mask: バイナリマスク（額・頬領域）
            landmarks: 468個のランドマーク座標
        """
        # 顔のランドマークを検出
        landmarks = self.detect_face(frame)

        if landmarks is None:
            return None, None

        # ROIマスクを生成
        roi_mask = self.extract_roi_mask(frame, landmarks, use_forehead, use_cheeks)

        return roi_mask, landmarks

    def visualize_roi(
        self,
        frame: np.ndarray,
        roi_mask: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        ROIを画像上に可視化

        Args:
            frame: BGR画像（OpenCV形式）
            roi_mask: ROIマスク
            landmarks: ランドマーク座標（オプション）
            color: ROIの色 (B, G, R)
            alpha: 透明度（0.0-1.0）

        Returns:
            vis_frame: ROIが重ね合わせられた画像
        """
        vis_frame = frame.copy()

        # ROIマスクをオーバーレイ
        overlay = vis_frame.copy()
        overlay[roi_mask > 0] = color
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

        # ランドマークを描画（オプション）
        if landmarks is not None:
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(vis_frame, (x, y), 1, (255, 0, 0), -1)

        return vis_frame

    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
