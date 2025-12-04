# リモート心拍数測定システム (Remote Heart Rate Monitoring)

Webカメラ動画から人間の皮膚の色変化を検出し、非接触で心拍数を測定するPythonシステムです。

## 概要

このプロジェクトは、rPPG (remote photoplethysmography) 技術を使用して、カメラ映像から心拍数をリアルタイムで推定します。論文「Remote Heart Rate Sensing and Projection to Renew Traditional Board Games and Foster Social Interactions」のアイデアを基に実装されています。

### 主要機能

- 🎥 **Webカメラからのリアルタイム心拍数測定**
- 👤 **MediaPipeによる高精度顔検出**
- 📊 **複数のrPPGアルゴリズム実装**
  - GREEN法（最もシンプル）
  - CHROM法（実用的精度）
  - POS法（最高精度）
  - ICA法（独立成分分析）
- 📈 **リアルタイム可視化**
- 🔬 **連続ウェーブレット変換による瞬時心拍数推定**

## セットアップ

### 必要な環境

- Python 3.8以上
- Webカメラ

### インストール

```bash
# リポジトリのクローン
cd remoteHeartRate

# 仮想環境の作成と依存ライブラリのインストール
uv sync
```

## 使用方法

### 基本的な使い方

```bash
# Webカメラからリアルタイム測定
uv run rppg-webcam

# 動画ファイルから測定
uv run rppg-video --input path/to/video.mp4

# オプション指定の例
uv run rppg-webcam --algorithm green --buffer-size 300 --fps 30

# ヘルプを表示
uv run rppg-webcam --help
```

### Pythonコードでの使用例

```python
from src.main import HeartRateMonitor

# モニターの初期化（CHROM法を使用）
monitor = HeartRateMonitor(algorithm='chrom', buffer_size=300)

# Webカメラから測定開始
monitor.run_webcam()
```

### アルゴリズムの選択

```python
# GREEN法（最もシンプル、学習用）
monitor = HeartRateMonitor(algorithm='green')

# CHROM法（推奨、バランスの良い精度と速度）
monitor = HeartRateMonitor(algorithm='chrom')

# POS法（最高精度、照明変化に強い）
monitor = HeartRateMonitor(algorithm='pos')

# ICA法（独立成分分析）
monitor = HeartRateMonitor(algorithm='ica')
```

## プロジェクト構造

```
remoteHeartRate/
├── README.md                    # このファイル
├── requirements.txt             # 依存ライブラリ
├── docs/
│   └── IMPLEMENTATION_PLAN.md   # 詳細な実装計画書
├── src/
│   ├── core/                   # 顔検出、信号抽出
│   ├── algorithms/             # rPPGアルゴリズム
│   ├── signal_processing/      # フィルタ、周波数解析
│   ├── utils/                  # ユーティリティ
│   └── main.py                 # メインアプリケーション
├── tests/                      # テストコード
├── examples/                   # サンプルスクリプト
└── data/                       # テストデータ、結果保存
```

## 技術詳細

### アルゴリズムの仕組み

1. **顔検出**: MediaPipe Face Meshで468個のランドマークを検出
2. **ROI抽出**: 額と頬領域から皮膚色の変化を抽出
3. **信号処理**: RGB信号をバンドパスフィルタで処理（0.75-3.5 Hz = 45-210 BPM）
4. **rPPG処理**: CHROM/POS法などで血液容積脈波（BVP）信号を抽出
5. **心拍数推定**: FFTまたはウェーブレット変換で周波数解析

### 推奨される測定条件

- ✅ 明るい照明（自然光または均一な室内照明）
- ✅ 顔全体がカメラに映っている
- ✅ 頭部を比較的安定させる
- ✅ カメラから50-100cm程度の距離
- ❌ 過度な動き、表情の変化を避ける
- ❌ 逆光、極端な照明変化を避ける

## 実装計画

詳細な実装計画は [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) を参照してください。

### 実装フェーズ

- **フェーズ1 (5日)**: MVP - 基本的な心拍数測定
- **フェーズ2 (4日)**: 精度向上 - CHROM/POS法実装
- **フェーズ3 (3日)**: ロバスト性向上 - GUI、動きアーティファクト除去
- **フェーズ4 (7日)**: 高度な機能 - 深層学習、複数人測定（オプション）

## テスト

```bash
# 単体テスト実行
uv run pytest tests/ -v

# 特定のテストのみ実行
uv run pytest tests/test_basic.py -v
```

## 参考文献

1. **Bousefsaf et al. (2013)** - "Continuous wavelet filtering on webcam photoplethysmographic signals"
2. **De Haan & Jeanne (2013)** - "Robust Pulse Rate From Chrominance-Based rPPG"
3. **Wang et al. (2016)** - "Algorithmic Principles of Remote PPG"

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更を行う場合は、まずissueを開いて変更内容を議論してください。

## 注意事項

このシステムは研究・教育目的です。医療用途には使用しないでください。正確な心拍数測定が必要な場合は、医療機器を使用してください。
