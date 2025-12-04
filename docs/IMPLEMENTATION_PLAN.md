# リモート心拍数測定システム実装計画

## 目次
1. [論文から学んだ主要な手法](#1-論文から学んだ主要な手法)
2. [rPPG技術の最新動向](#2-rppg技術の最新動向)
3. [必要なライブラリ](#3-必要なライブラリ)
4. [詳細な実装ステップ](#4-詳細な実装ステップ)
5. [予想される課題と解決策](#5-予想される課題と解決策)
6. [テスト方法](#6-テスト方法)
7. [実装スケジュール](#7-実装スケジュール)
8. [推奨される実装アプローチ](#8-推奨される実装アプローチ)
9. [参考文献・リソース](#9-参考文献リソース)
10. [ファイル構成](#10-ファイル構成)

---

## 1. 論文から学んだ主要な手法

### 1.1 論文の核心的知見

**使用されたアルゴリズム:**
- **Frédéric Bousefsaf et al. (2013)の連続ウェーブレットフィルタリング手法**
  - Webカメラのフォトプレチスモグラフィック信号に連続ウェーブレット変換を適用
  - 瞬時心拍数をリモート評価
  - ECGとの相関: 0.30 ~ 0.81（平均0.53）

**技術的アプローチ:**
- 顔全体の平均値を使用（計算効率重視）
- RGB信号から皮膚色の微細な変化を検出
- 緑チャンネルが最も強い脈動信号を提供（ヘモグロビンの緑光吸収特性）

**ハードウェア仕様:**
- Sony PlayStation Eye カメラ（低コスト：約10ドル）
- 解像度: 640×480、30 FPS
- 生のBayerマトリクスデータを取得

**検証結果:**
- 10分間セッション×3で検証
- 深呼吸による心拍変動も検出可能
- 有酸素運動後の測定でも機能

### 1.2 システム設計の重要ポイント

1. **非接触センシング**: ウェアラブルセンサー不要
2. **リアルタイム処理**: 3人同時測定を1台のコンピュータで実現
3. **シームレスな統合**: プロジェクションマッピングで技術を隠蔽

---

## 2. rPPG技術の最新動向

### 2.1 主要アルゴリズム

**GREEN法（最もシンプル）:**
- 緑チャンネルのみを使用
- 実装が容易で計算コスト低
- 基本的なベースライン手法

**CHROM法（Chrominance-based, 2013）:**
- YCbCr色空間のCb（青差分）とCr（赤差分）チャンネルを使用
- 色度と輝度を分離して動きアーティファクトを軽減
- CIELab色空間での実装も効果的

**POS法（Plane-Orthogonal-to-Skin, 2016）:**
- RGB信号の線形結合で皮膚反射に直交する平面を構築
- 照明変化に対するロバスト性が高い
- CHROMより高精度

**ICA/PCA法（Blind Source Separation）:**
- 独立成分分析/主成分分析
- 教師なし学習
- ICAの第2成分に通常強いrPPG信号が含まれる

**深層学習ベース（MTTS-CAN など）:**
- エンドツーエンド学習
- MAE: 6.0 bpm（明るい肌）～ 9.5 bpm（暗い肌）
- より高精度だが計算コスト高

### 2.2 推奨アルゴリズム（段階的実装）

**フェーズ1: GREEN + バンドパスフィルタ**
- 実装の容易さで学習に最適
- 基本的なパイプラインの確立

**フェーズ2: CHROM**
- 動き耐性の向上
- 実用的な精度

**フェーズ3: POS**
- 照明変化への対応
- 最高の信号品質（古典的手法中）

**フェーズ4（オプション）: 深層学習**
- pyVHRのMTTS-CANモデル利用
- より高精度な測定

---

## 3. 必要なライブラリ

### 3.1 コアライブラリ

```
opencv-python>=4.8.0        # 動画処理、顔検出
mediapipe>=0.10.8          # 顔ランドマーク検出（推奨）
numpy>=1.24.0              # 数値計算
scipy>=1.10.0              # 信号処理（フィルタ、周波数解析）
```

### 3.2 信号処理・解析

```
PyWavelets>=1.4.1          # 連続ウェーブレット変換（論文手法）
scikit-learn>=1.3.0        # ICA/PCA実装
```

### 3.3 可視化・UI

```
matplotlib>=3.7.0          # グラフ表示
```

### 3.4 オプション（高度な実装）

```
pyVHR                      # rPPG統合フレームワーク
dlib                       # 代替顔検出（ただしMediaPipeより低速）
```

### 3.5 顔検出ライブラリの比較結果

| 手法 | 精度 | 速度 | 推奨度 |
|------|------|------|--------|
| MediaPipe | 98.6% | 最速 | ★★★★★ |
| OpenCV DNN | 良好 | 中速 | ★★★★ |
| dlib HoG | 89% | 遅い | ★★ |
| Haar Cascade | 84.5% | 速い | ★ |

**結論: MediaPipe Face Meshを使用**

---

## 4. 詳細な実装ステップ

### 4.1 プロジェクト構造

```
remoteHeartRate/
├── README.md                    # プロジェクト概要
├── requirements.txt             # 依存ライブラリ
├── docs/
│   └── IMPLEMENTATION_PLAN.md   # この実装計画書
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── face_detector.py    # 顔検出・ROI抽出
│   │   ├── signal_extractor.py # RGB信号抽出
│   │   └── roi_manager.py      # ROI管理
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── green.py            # GREEN法
│   │   ├── chrom.py            # CHROM法
│   │   ├── pos.py              # POS法
│   │   ├── ica.py              # ICA法
│   │   └── base.py             # 基底クラス
│   ├── signal_processing/
│   │   ├── __init__.py
│   │   ├── filters.py          # バンドパスフィルタ
│   │   ├── frequency_analysis.py # FFT、ウェーブレット
│   │   └── heart_rate_estimator.py # BPM推定
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── video_io.py         # 動画入出力
│   │   ├── visualization.py    # リアルタイム可視化
│   │   └── validation.py       # 検証ツール
│   └── main.py                 # メインアプリケーション
├── tests/
│   ├── __init__.py
│   ├── test_face_detector.py
│   ├── test_signal_extractor.py
│   └── test_algorithms.py
├── examples/
│   ├── webcam_demo.py          # Webカメラデモ
│   ├── video_file_demo.py      # 動画ファイル処理
│   └── comparison_demo.py      # アルゴリズム比較
└── data/
    ├── test_videos/            # テスト用動画
    └── results/                # 結果保存
```

### 4.2 実装ステップ（段階的アプローチ）

#### **ステップ1: 環境セットアップ（Day 1）**

```bash
# 仮想環境作成
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 基本ライブラリインストール
pip install opencv-python mediapipe numpy scipy matplotlib PyWavelets
```

#### **ステップ2: 顔検出とROI抽出（Day 1-2）**

**実装内容:**
1. MediaPipe Face Meshによる顔検出
2. 468個のランドマークから顔下半分のROI定義
3. 額・頬領域の特定（強い脈動信号）
4. ROIマスク生成

**キーコード:**
```python
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_roi(self, frame):
        """
        顔のROIを検出

        Returns:
            roi_mask: バイナリマスク（額・頬領域）
            landmarks: 468個のランドマーク座標
        """
        # 実装詳細...
```

**ROI選択戦略:**
- 額: ランドマーク10, 338, 297, 332, 284, 251, 389, 356
- 左頬: ランドマーク50, 36, 206, 216, 212, 202
- 右頬: ランドマーク280, 266, 426, 436, 432, 422

#### **ステップ3: RGB信号抽出（Day 2-3）**

**実装内容:**
1. ROI内のピクセル平均値計算（RGB各チャンネル）
2. 時系列データの蓄積（バッファ管理）
3. 正規化処理

**キーコード:**
```python
def extract_rgb_signals(frame, roi_mask):
    """
    ROIからRGB信号を抽出

    Args:
        frame: BGR画像（OpenCV形式）
        roi_mask: ROIマスク

    Returns:
        (r_mean, g_mean, b_mean): RGB平均値
    """
    # ROI領域のみを抽出
    roi_pixels = frame[roi_mask > 0]

    # BGR → RGB変換して平均値計算
    b_mean = np.mean(roi_pixels[:, 0])
    g_mean = np.mean(roi_pixels[:, 1])
    r_mean = np.mean(roi_pixels[:, 2])

    return r_mean, g_mean, b_mean
```

#### **ステップ4: 信号処理フィルタ（Day 3-4）**

**実装内容:**
1. バンドパスフィルタ（0.75 Hz - 3.5 Hz = 45-210 BPM）
2. Butterworthフィルタ（4次）
3. デトレンディング（線形トレンド除去）

**キーコード:**
```python
from scipy.signal import butter, sosfiltfilt, detrend

def bandpass_filter(signal, lowcut=0.75, highcut=3.5, fs=30, order=4):
    """
    バンドパスフィルタ適用

    Args:
        signal: 入力信号
        lowcut: 下限周波数 (Hz)
        highcut: 上限周波数 (Hz)
        fs: サンプリング周波数 (FPS)
        order: フィルタ次数

    Returns:
        filtered_signal: フィルタ後の信号
    """
    # デトレンディング
    signal = detrend(signal, type='linear')

    # Butterworthフィルタ設計
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

    # ゼロ位相フィルタリング
    filtered = sosfiltfilt(sos, signal)

    return filtered
```

#### **ステップ5: アルゴリズム実装（Day 4-7）**

**5.1 GREEN法（最もシンプル）**

```python
def green_method(rgb_signals):
    """
    GREEN法: 緑チャンネルのみ使用

    Args:
        rgb_signals: (r_signal, g_signal, b_signal)

    Returns:
        bvp_signal: 血液容積脈波信号
    """
    _, g_signal, _ = rgb_signals

    # 正規化
    bvp_signal = (g_signal - np.mean(g_signal)) / np.std(g_signal)

    return bvp_signal
```

**5.2 CHROM法（論文推奨）**

```python
def chrom_method(rgb_signals):
    """
    CHROM法: 色度ベース

    参考: G. de Haan and V. Jeanne (2013)
    """
    r_signal, g_signal, b_signal = rgb_signals

    # 正規化
    r_norm = r_signal / np.mean(r_signal)
    g_norm = g_signal / np.mean(g_signal)
    b_norm = b_signal / np.mean(b_signal)

    # 色度信号計算
    X = 3 * r_norm - 2 * g_norm
    Y = 1.5 * r_norm + g_norm - 1.5 * b_norm

    # 直交化
    alpha = np.std(X) / np.std(Y)
    bvp_signal = X - alpha * Y

    return bvp_signal
```

**5.3 POS法（最高精度）**

```python
def pos_method(rgb_signals, window_size=30):
    """
    POS法: Plane-Orthogonal-to-Skin

    参考: W. Wang et al. (2016)
    """
    r_signal, g_signal, b_signal = rgb_signals

    # RGB信号を行列に変換
    C = np.array([r_signal, g_signal, b_signal])

    # 正規化
    C_norm = C / np.mean(C, axis=1, keepdims=True)

    # スライディングウィンドウで処理
    bvp_signal = []
    for i in range(0, len(r_signal) - window_size, 1):
        window = C_norm[:, i:i+window_size]

        # 平面直交ベクトル計算
        S1 = window[0, :] - window[1, :]
        S2 = window[0, :] + window[1, :] - 2 * window[2, :]

        # BVP信号
        h = S1 + (np.std(S1) / np.std(S2)) * S2
        bvp_signal.append(np.mean(h))

    return np.array(bvp_signal)
```

**5.4 ICA法**

```python
from sklearn.decomposition import FastICA

def ica_method(rgb_signals, n_components=3):
    """
    ICA法: 独立成分分析
    """
    # RGB信号を行列に変換
    X = np.array(rgb_signals).T

    # ICA適用
    ica = FastICA(n_components=n_components, random_state=42)
    S = ica.fit_transform(X)

    # 第2成分を使用（通常、強いrPPG信号を含む）
    bvp_signal = S[:, 1]

    return bvp_signal
```

#### **ステップ6: 心拍数推定（Day 7-8）**

**6.1 FFTベース（高速）**

```python
def estimate_heart_rate_fft(bvp_signal, fs=30):
    """
    FFTによる心拍数推定

    Args:
        bvp_signal: BVP信号
        fs: サンプリング周波数

    Returns:
        heart_rate: 推定心拍数（BPM）
        confidence: 信頼度スコア
    """
    # FFT計算
    N = len(bvp_signal)
    fft_result = np.fft.rfft(bvp_signal)
    freqs = np.fft.rfftfreq(N, 1/fs)

    # 生理学的範囲でマスク（0.75-3.5 Hz = 45-210 BPM）
    mask = (freqs >= 0.75) & (freqs <= 3.5)
    freqs_masked = freqs[mask]
    fft_masked = np.abs(fft_result[mask])

    # ピーク周波数検出
    peak_idx = np.argmax(fft_masked)
    peak_freq = freqs_masked[peak_idx]

    # BPM変換
    heart_rate = peak_freq * 60

    # 信頼度（ピークの突出度）
    confidence = fft_masked[peak_idx] / np.mean(fft_masked)

    return heart_rate, confidence
```

**6.2 連続ウェーブレット変換（論文手法）**

```python
import pywt

def estimate_heart_rate_cwt(bvp_signal, fs=30):
    """
    連続ウェーブレット変換による心拍数推定

    論文手法: Bousefsaf et al. (2013)
    """
    # スケール範囲設定（対応周波数: 0.75-3.5 Hz）
    scales = pywt.scale2frequency('morl', np.arange(1, 128)) * fs
    scales_mask = (scales >= 0.75) & (scales <= 3.5)
    scales = np.arange(1, 128)[scales_mask]

    # CWT計算
    coefficients, frequencies = pywt.cwt(bvp_signal, scales, 'morl', 1/fs)

    # 各時刻の主要周波数を検出
    power = np.abs(coefficients) ** 2
    dominant_freq_idx = np.argmax(power, axis=0)

    # 瞬時心拍数計算
    instantaneous_hr = frequencies[dominant_freq_idx] * 60

    # 平均心拍数
    heart_rate = np.median(instantaneous_hr)

    return heart_rate, instantaneous_hr
```

**6.3 自己相関法（ロバスト）**

```python
def estimate_heart_rate_autocorr(bvp_signal, fs=30):
    """
    自己相関による心拍数推定
    """
    # 自己相関計算
    autocorr = np.correlate(bvp_signal, bvp_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # 生理学的範囲のラグ（45-210 BPM）
    min_lag = int(fs * 60 / 210)  # 210 BPMに対応
    max_lag = int(fs * 60 / 45)   # 45 BPMに対応

    # 範囲内でピーク検出
    autocorr_range = autocorr[min_lag:max_lag]
    peak_lag = np.argmax(autocorr_range) + min_lag

    # BPM計算
    heart_rate = 60 * fs / peak_lag

    return heart_rate
```

#### **ステップ7: リアルタイム可視化（Day 8-9）**

**実装内容:**
1. リアルタイムグラフ表示（BVP信号、周波数スペクトル）
2. 心拍数表示（BPM、信頼度）
3. ROI可視化

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HeartRateVisualizer:
    def __init__(self):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))

    def update(self, frame_data):
        """
        リアルタイム更新

        Args:
            frame_data: {
                'bvp_signal': BVP信号,
                'frequencies': 周波数配列,
                'spectrum': スペクトル,
                'heart_rate': 推定BPM,
                'confidence': 信頼度
            }
        """
        # BVP信号プロット
        self.axes[0].clear()
        self.axes[0].plot(frame_data['bvp_signal'])
        self.axes[0].set_title('BVP Signal')

        # 周波数スペクトル
        self.axes[1].clear()
        self.axes[1].plot(frame_data['frequencies'], frame_data['spectrum'])
        self.axes[1].set_title('Frequency Spectrum')

        # 心拍数表示
        self.axes[2].clear()
        self.axes[2].text(0.5, 0.5,
                         f"Heart Rate: {frame_data['heart_rate']:.1f} BPM\n"
                         f"Confidence: {frame_data['confidence']:.2f}",
                         ha='center', va='center', fontsize=20)
```

#### **ステップ8: メインアプリケーション（Day 9-10）**

```python
from collections import deque

class HeartRateMonitor:
    def __init__(self, algorithm='chrom', buffer_size=300):
        """
        Args:
            algorithm: 'green', 'chrom', 'pos', 'ica'
            buffer_size: 信号バッファサイズ（秒×FPS）
        """
        self.face_detector = FaceDetector()
        self.algorithm = algorithm
        self.buffer_size = buffer_size

        # 信号バッファ
        self.rgb_buffer = {
            'r': deque(maxlen=buffer_size),
            'g': deque(maxlen=buffer_size),
            'b': deque(maxlen=buffer_size)
        }

    def process_frame(self, frame):
        """
        フレーム処理
        """
        # 1. 顔検出・ROI抽出
        roi_mask, landmarks = self.face_detector.detect_roi(frame)

        if roi_mask is None:
            return None

        # 2. RGB信号抽出
        r, g, b = extract_rgb_signals(frame, roi_mask)
        self.rgb_buffer['r'].append(r)
        self.rgb_buffer['g'].append(g)
        self.rgb_buffer['b'].append(b)

        # バッファが十分溜まったら処理
        if len(self.rgb_buffer['r']) < self.buffer_size:
            return None

        # 3. 信号処理
        rgb_signals = (
            np.array(self.rgb_buffer['r']),
            np.array(self.rgb_buffer['g']),
            np.array(self.rgb_buffer['b'])
        )

        # 4. アルゴリズム適用
        if self.algorithm == 'green':
            bvp = green_method(rgb_signals)
        elif self.algorithm == 'chrom':
            bvp = chrom_method(rgb_signals)
        elif self.algorithm == 'pos':
            bvp = pos_method(rgb_signals)
        elif self.algorithm == 'ica':
            bvp = ica_method(rgb_signals)

        # 5. バンドパスフィルタ
        bvp_filtered = bandpass_filter(bvp, fs=30)

        # 6. 心拍数推定
        hr, confidence = estimate_heart_rate_fft(bvp_filtered, fs=30)

        return {
            'heart_rate': hr,
            'confidence': confidence,
            'bvp_signal': bvp_filtered,
            'roi_mask': roi_mask
        }

    def run_webcam(self):
        """
        Webカメラからリアルタイム測定
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            if result:
                # 結果を画面に表示
                cv2.putText(frame,
                           f"HR: {result['heart_rate']:.1f} BPM",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

                # ROI可視化
                frame[result['roi_mask'] > 0] = [0, 255, 0]

            cv2.imshow('Heart Rate Monitor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
```

---

## 5. 予想される課題と解決策

### 5.1 課題1: 動きアーティファクト

**問題:**
- 頭部の動き、表情変化がノイズとなる

**解決策:**
1. **顔トラッキングの改善**
   - MediaPipeの高精度ランドマーク追跡
   - 安定化アルゴリズム（Kalmanフィルタ）

2. **アルゴリズム選択**
   - CHROM/POS法は動きに対してロバスト
   - CIELab色空間使用（色度と輝度分離）

3. **信号処理**
   - 適応フィルタ
   - ウェーブレット変換（非定常信号に強い）

### 5.2 課題2: 照明変化

**問題:**
- 環境光の変動がRGB信号に影響

**解決策:**
1. **正規化**
   - 時間軸正規化（各チャンネルを平均で除算）
   - 空間正規化（複数ROIの平均）

2. **POS法採用**
   - 照明変化に直交する平面を使用
   - 照明不変性が高い

3. **前処理**
   - ヒストグラム均等化
   - ホワイトバランス調整

### 5.3 課題3: 肌色バイアス

**問題:**
- 暗い肌色でMAEが増加（5.2 → 14.1 BPM）

**解決策:**
1. **データ拡張**
   - PhysFlowなどの肌色変換技術
   - 多様な肌色でのテスト

2. **カメラ設定**
   - ダイナミックレンジ拡大
   - HDRモード使用

3. **アルゴリズム**
   - 深層学習モデル（肌色バイアス小）
   - マルチモーダル融合

### 5.4 課題4: リアルタイム性能

**問題:**
- 処理速度とフレームレートの両立

**解決策:**
1. **最適化**
   - NumPyベクトル化
   - C++拡張（必要に応じて）

2. **バッファ管理**
   - スライディングウィンドウ
   - インクリメンタル計算

3. **マルチスレッド**
   - フレーム取得と信号処理の分離
   - `threading`または`multiprocessing`使用

### 5.5 課題5: 信号品質評価

**問題:**
- 推定の信頼性が不明

**解決策:**
1. **信頼度スコア算出**
   - SNR（信号対雑音比）
   - ピークの突出度
   - 周波数スペクトルの鋭さ

2. **品質閾値**
   - 低信頼度時は推定を表示しない
   - ユーザーへのフィードバック（「顔を安定させてください」）

3. **時間平均**
   - 移動平均フィルタ
   - 外れ値除去

---

## 6. テスト方法

### 6.1 単体テスト

```python
# tests/test_face_detector.py
def test_face_detection():
    detector = FaceDetector()
    test_image = cv2.imread('data/test_images/face.jpg')
    roi_mask, landmarks = detector.detect_roi(test_image)
    assert roi_mask is not None
    assert len(landmarks) == 468

# tests/test_algorithms.py
def test_chrom_method():
    # 合成信号でテスト
    t = np.linspace(0, 10, 300)
    hr_true = 72  # BPM
    freq_true = hr_true / 60  # Hz

    # 合成RGB信号（心拍数72 BPM）
    r_signal = np.sin(2 * np.pi * freq_true * t) + np.random.normal(0, 0.1, len(t))
    g_signal = 1.2 * np.sin(2 * np.pi * freq_true * t) + np.random.normal(0, 0.1, len(t))
    b_signal = 0.8 * np.sin(2 * np.pi * freq_true * t) + np.random.normal(0, 0.1, len(t))

    bvp = chrom_method((r_signal, g_signal, b_signal))
    hr_estimated, _ = estimate_heart_rate_fft(bvp, fs=30)

    assert abs(hr_estimated - hr_true) < 5  # 誤差5 BPM以内
```

### 6.2 統合テスト

**シナリオ:**
1. 既知の心拍数動画でテスト
2. 接触式心拍計（グランドトゥルース）との比較
3. 異なる条件でのテスト:
   - 照明変化
   - 頭部動き
   - 異なる肌色

**評価指標:**
- MAE（平均絶対誤差）
- RMSE（二乗平均平方根誤差）
- Pearson相関係数
- Bland-Altmanプロット

### 6.3 ベンチマークデータセット

公開データセット使用:
- **PURE Dataset**: 10人、様々な頭部動き
- **UBFC-rPPG**: 42人、RGB動画
- **COHFACE**: 40人、様々な照明条件

---

## 7. 実装スケジュール

| 期間 | タスク | 成果物 |
|------|--------|--------|
| Day 1 | 環境セットアップ、プロジェクト構造作成 | requirements.txt, 基本ディレクトリ |
| Day 2-3 | 顔検出・ROI抽出実装 | face_detector.py, roi_manager.py |
| Day 3-4 | 信号抽出・フィルタ実装 | signal_extractor.py, filters.py |
| Day 4-5 | GREEN/CHROM法実装 | green.py, chrom.py |
| Day 6 | POS/ICA法実装 | pos.py, ica.py |
| Day 7-8 | 心拍数推定実装 | heart_rate_estimator.py |
| Day 8-9 | 可視化実装 | visualization.py |
| Day 9-10 | メインアプリケーション | main.py, webcam_demo.py |
| Day 11-12 | テスト・デバッグ | tests/ |
| Day 13-14 | ドキュメント・最適化 | README.md, docs/ |

---

## 8. 推奨される実装アプローチ

### 8.1 フェーズ1: MVP（Minimum Viable Product）

**目標**: 基本的な心拍数測定を実現

**含まれる機能:**
- MediaPipeによる顔検出
- GREEN法による信号抽出
- FFTによる心拍数推定
- Webカメラからのリアルタイム処理

**期間**: 5日間

### 8.2 フェーズ2: 精度向上

**目標**: より正確な測定

**追加機能:**
- CHROM/POS法実装
- バンドパスフィルタ最適化
- 複数ROI統合
- 信頼度スコア

**期間**: 4日間

### 8.3 フェーズ3: ロバスト性向上

**目標**: 実用的なシステム

**追加機能:**
- 動きアーティファクト除去
- 照明補正
- GUI実装
- 結果保存機能

**期間**: 3日間

### 8.4 フェーズ4: 高度な機能（オプション）

**追加機能:**
- 連続ウェーブレット変換（論文手法）
- ICA/PCA実装
- 複数人同時測定
- 深層学習モデル統合（pyVHR）

**期間**: 7日間

---

## 9. 参考文献・リソース

### 9.1 主要論文

1. **Bousefsaf et al. (2013)** - 論文で使用された手法
   - "Continuous wavelet filtering on webcam photoplethysmographic signals"

2. **De Haan & Jeanne (2013)** - CHROM法
   - "Robust Pulse Rate From Chrominance-Based rPPG"

3. **Wang et al. (2016)** - POS法
   - "Algorithmic Principles of Remote PPG"

### 9.2 実装リソース

- **pyVHR GitHub**: https://github.com/phuselab/pyVHR
- **rppg-pos GitHub**: https://github.com/pavisj/rppg-pos
- **yarppg Tutorial**: https://www.samproell.io/posts/yarppg/

### 9.3 公開データセット

- PURE Dataset
- UBFC-rPPG
- COHFACE
- MAHNOB-HCI

---

## 10. ファイル構成

```
remoteHeartRate/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── IMPLEMENTATION_PLAN.md (この計画書)
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── face_detector.py       # MediaPipe顔検出
│   │   ├── signal_extractor.py    # RGB信号抽出
│   │   └── roi_manager.py         # ROI管理
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py               # 基底クラス
│   │   ├── green.py              # GREEN法
│   │   ├── chrom.py              # CHROM法
│   │   ├── pos.py                # POS法
│   │   └── ica.py                # ICA法
│   ├── signal_processing/
│   │   ├── __init__.py
│   │   ├── filters.py            # バンドパスフィルタ
│   │   ├── frequency_analysis.py # FFT、ウェーブレット
│   │   └── heart_rate_estimator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── video_io.py          # 動画入出力
│   │   ├── visualization.py     # 可視化
│   │   └── validation.py        # 検証ツール
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_face_detector.py
│   ├── test_signal_extractor.py
│   └── test_algorithms.py
├── examples/
│   ├── webcam_demo.py           # Webカメラデモ
│   ├── video_file_demo.py       # 動画ファイル処理
│   └── comparison_demo.py       # アルゴリズム比較
└── data/
    ├── test_videos/
    ├── test_images/
    └── results/
```

---

## まとめ

この実装計画は、論文の知見（連続ウェーブレット変換）と最新のrPPG技術（CHROM, POS法）を統合し、段階的に実装できるよう設計されています。各ステップは独立してテスト可能で、必要に応じて拡張できる柔軟な構造となっています。

**推奨される開始方法:**
1. フェーズ1（MVP）から開始
2. GREEN法で基本パイプラインを確立
3. 動作確認後、CHROM法で精度向上
4. 必要に応じてPOS法、深層学習へ拡張
