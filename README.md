# pamiq-vision-exp

Continuous Learning 環境下で Vision 表現学習手法を実験・評価するフレームワーク。

Continuous Learning はタスク境界を持たないストリーミングデータからのオンライン学習を扱う枠組みであり、明示的なタスク切り替えを前提とする Continual Learning とは異なる。データ分布は時間とともに非定常に変化し、学習は FIFO リプレイバッファを用いたストリーミング処理で進行する。

本フレームワークはエゴセントリックなロボット映像ストリームを入力として、異なる表現学習手法（V-JEPA, VAE）を同一条件で公平に比較する。評価は以下の 2 軸で行う:

- **入力再構成**: エンコーダ特徴からの入力復元精度を測定し、情報保存性を評価する
- **未来予測**: エンコーダ特徴からの将来フレーム予測精度を測定し、ダイナミクスの近似能力を評価する

再構成のみでは過完備な表現を検出できず、予測のみでは表現の崩壊を検出できないため、両軸を併用する。

## 技術スタック

- Python 3.13+
- PyTorch 2.9+, torchvision
- Hydra（設定管理）
- ClearML（実験管理、オプション）
- uv（パッケージ管理）
- pytest, pyright, ruff（開発ツール）

## セットアップ

前提条件: Python 3.13 以上、uv がインストール済みであること。

```bash
make setup
```

仮想環境の作成、依存パッケージのインストール、pre-commit フックの設定を行う。

### 開発コマンド

| コマンド         | 説明                              |
| ---------------- | --------------------------------- |
| `make format`    | フォーマッタの実行                |
| `make type`      | 型チェックの実行                  |
| `make test`      | テストの実行（slow テストを除く） |
| `make test-slow` | 全テストの実行                    |

### Docker

```bash
make docker-build    # イメージのビルド
make docker-up       # コンテナの起動（GPU を自動検出）
make docker-attach   # 起動中のコンテナにアタッチ
make docker-down     # コンテナの停止
```

## 学習

基本コマンド:

```bash
python src/train.py data.frame_loader.video_list_path=path/to/videos.txt
```

実験設定を指定して実行:

```bash
# V-JEPA（軽量設定）
python src/train.py experiment=vjepa_short data.frame_loader.video_list_path=path/to/videos.txt

# VAE
python src/train.py experiment=vae_short data.frame_loader.video_list_path=path/to/videos.txt
```

### 主な設定グループ

| 設定キー     | 説明               | デフォルト設定                  |
| ------------ | ------------------ | ------------------------------- |
| `model`      | 表現学習モデル     | `configs/model/vjepa.yaml`      |
| `data`       | データパイプライン | `configs/data/default.yaml`     |
| `training`   | 学習ループ         | `configs/training/default.yaml` |
| `experiment` | 実験設定（上書き） | `configs/experiment/`           |

チェックポイントは `logs/<experiment_name>/<timestamp>/checkpoints/` に保存される。

ClearML が利用可能な場合、実験管理が自動的に有効になる。

## 評価

基本コマンド:

```bash
python src/eval.py \
  checkpoint_path=logs/.../checkpoints/300.ckpt \
  data.frame_loader.video_list_path=path/to/videos.txt
```

`checkpoint_path` は必須パラメータである。特定の評価をスキップするには null を指定する（例: `evaluation.prediction=null`）。

### 評価タイプ

- **入力再構成**: 凍結したエンコーダ特徴上に軽量 CNN デコーダを学習し、入力画像の復元精度を測定する。メトリクス: MAE, MSE。
- **未来予測**: 凍結したエンコーダ特徴上に MinGRU を学習し、将来の特徴量を予測する。予測ホライズン: 1, 2, 4, 8 ステップ。メトリクス: ホライズンごとの MAE。

`encoder_key` のデフォルトは `context_encoder`（V-JEPA 用）である。VAE を評価する場合は `encoder_key=encoder` を指定する。

## プロジェクト構成

```
pamiq-vision-exp/
├── src/
│   ├── train.py              # 学習エントリーポイント
│   ├── eval.py               # 評価エントリーポイント
│   └── exp/                  # メインパッケージ
│       ├── data/             # データパイプライン
│       ├── models/           # 表現学習モデル（V-JEPA, VAE）
│       ├── trainers/         # 手法別学習ロジック
│       ├── training/         # 学習ループ・チェックポイント管理
│       └── evaluation/       # 評価（再構成・未来予測・ベースライン）
├── configs/                  # Hydra 設定ファイル
│   ├── train.yaml
│   ├── eval.yaml
│   ├── model/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── experiment/           # 実験設定
├── tests/                    # テストスイート
├── docs/                     # 要件定義ドキュメント
├── data/                     # 動画ファイル（ユーザーが配置）
└── logs/                     # チェックポイント出力先（git 管理外）
```

## 設定のカスタマイズ

設定は Hydra ベースの YAML 階層構造で管理されており、CLI からのオーバーライドが可能である。

実験固有の設定は `configs/experiment/` に YAML ファイルとして作成し、`experiment=<name>` で指定する。既存の実験設定として `vjepa_short` と `vae_short` が用意されている。

## 要件定義

詳細な要件定義ドキュメントは `docs/` ディレクトリで管理している。

[docs/README.md](docs/README.md) を参照。
