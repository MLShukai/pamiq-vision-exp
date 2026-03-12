# 表現学習手法

## 共通インターフェース

すべての表現学習手法は以下の共通仕様に従う。手法は置換可能な設計とする。

### エンコーダ

- **入力**: スタッキングされた動画フレーム `[batch, channels, time, height, width]`
- **出力**: 低次元の特徴量ベクトル

出力の形状は手法によって異なることを許容する。ただし先頭次元は必ずバッチ次元とし、`(batch, *)` の形式に従う。

### 学習に必要なコンポーネント

手法によってはエンコーダ以外のコンポーネント（予測器、ターゲットエンコーダ等）を持つ。これらは学習時にのみ使用し、評価時にはエンコーダのみを凍結して利用する。

## V-JEPA

### 概要

Video Joint Embedding Predictive Architecture。V-JEPA2 の設計に基づき、3D RoPE を使用する。マスクされた動画パッチの潜在表現を予測することで、ピクセル空間のデコーダを必要とせずに表現を学習する。

### アーキテクチャ

```
動画入力 [B, C, T, H, W]
    ↓
VideoPatchifier (3D Conv)
    ↓
Tubelet 埋め込み [B, n_tubelets, hidden_dim]
    ↓
┌──────────────────────┬──────────────────────┐
│  Context Encoder     │  Target Encoder      │
│  (マスクあり)         │  (マスクなし、EMA)    │
│  Transformer + RoPE  │  Transformer + RoPE  │
│         ↓            │         ↓            │
│  [B, n, embed_dim]   │  [B, n, embed_dim]   │
└──────────┬───────────┴──────────┬───────────┘
           ↓                      ↓
      Predictor               教師信号
      (Transformer)
           ↓
    予測 [B, n, embed_dim]
           ↓
    損失: 予測 vs Target Encoder 出力
```

### コンポーネント

#### VideoPatchifier

- 3D 畳み込みで動画をチューブレット（時空間パッチ）に分割
- デフォルトのチューブレットサイズ: `(2, 16, 16)` — (temporal, height, width)
- 出力: `[batch, n_tubelets, hidden_dim]`

#### Context Encoder

- マスクされたチューブレットに対してマスクトークンを挿入
- Transformer + 3D RoPE で処理
- 出力を `embed_dim` に射影

#### Target Encoder

- Context Encoder の EMA（指数移動平均）コピー
- マスクなしで全チューブレットを処理
- 学習対象の教師信号を生成

#### Predictor

- Context Encoder の出力から Target Encoder の出力を予測
- 予測対象のチューブレット位置に予測トークンを加算
- Transformer + 3D RoPE で処理

### 主要パラメータ

| パラメータ     | 説明                                                 | デフォルト    |
| -------------- | ---------------------------------------------------- | ------------- |
| `tubelet_size` | チューブレットサイズ (T, H, W)                       | `(2, 16, 16)` |
| `hidden_dim`   | Transformer の隠れ次元                               | `768`         |
| `embed_dim`    | 出力埋め込み次元                                     | `128`         |
| `depth`        | Transformer 層数                                     | `6`           |
| `num_heads`    | アテンションヘッド数                                 | `12`          |
| `rope_theta`   | RoPE の基底周波数                                    | `10000.0`     |
| `ema_momentum` | Target Encoder の EMA 更新率（学習経過によらず一定） | `0.996`       |

## 今後の候補手法

実装の優先順位: VAE > MAE

| 優先度 | 手法                     | 種類         | 備考                         |
| ------ | ------------------------ | ------------ | ---------------------------- |
| 1      | VAE                      | 再構成ベース | ベースラインとしても使用     |
| 2      | MAE (Masked Autoencoder) | マスク再構成 | ピクセル空間での再構成を学習 |

### マスキング戦略

V-JEPA・MAE 等のマスクベース手法のマスキング戦略は、各手法の原論文に従う。
