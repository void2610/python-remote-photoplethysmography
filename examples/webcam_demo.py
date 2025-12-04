#!/usr/bin/env python3
"""
Webカメラデモ

Webカメラからリアルタイムで心拍数を測定します。
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import HeartRateMonitor


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Webカメラからリアルタイム心拍数測定'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='カメラデバイスID (デフォルト: 0)'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='green',
        choices=['green'],
        help='rPPGアルゴリズム (デフォルト: green)'
    )

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=300,
        help='信号バッファサイズ (デフォルト: 300フレーム = 10秒@30fps)'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='カメラFPS (デフォルト: 30.0)'
    )

    parser.add_argument(
        '--estimation-method',
        type=str,
        default='fft',
        choices=['fft', 'peaks', 'autocorr', 'cwt'],
        help='心拍数推定手法 (デフォルト: fft)'
    )

    parser.add_argument(
        '--save-video',
        type=str,
        default=None,
        help='動画保存パス (例: output.mp4)'
    )

    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='可視化を無効にする'
    )

    args = parser.parse_args()

    # モニター初期化
    print("=" * 60)
    print("リモート心拍数測定システム - Webカメラデモ")
    print("=" * 60)
    print(f"アルゴリズム: {args.algorithm.upper()}")
    print(f"バッファサイズ: {args.buffer_size}フレーム")
    print(f"FPS: {args.fps}")
    print(f"推定手法: {args.estimation_method}")
    print(f"カメラID: {args.camera}")
    if args.save_video:
        print(f"動画保存: {args.save_video}")
    print("=" * 60)
    print("\n操作方法:")
    print("  'q' キー: 終了")
    print("  'r' キー: バッファリセット")
    print("\n測定のヒント:")
    print("  - 顔全体がカメラに映るようにしてください")
    print("  - 明るい場所で測定してください")
    print("  - 頭をなるべく動かさないでください")
    print("  - バッファが溜まるまで約10秒かかります")
    print("=" * 60)
    print()

    monitor = HeartRateMonitor(
        algorithm=args.algorithm,
        buffer_size=args.buffer_size,
        fps=args.fps,
        estimation_method=args.estimation_method
    )

    try:
        monitor.run_webcam(
            camera_id=args.camera,
            show_visualization=not args.no_visualization,
            save_video=args.save_video
        )
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みにより終了します...")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n測定完了")


if __name__ == "__main__":
    main()
