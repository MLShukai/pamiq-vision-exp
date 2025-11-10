# PAMIQ Vision Experiments

画像のエンコーダのリアルタイム継続学習の実験リポジトリ

## 概要

このリポジトリは、PAMIQの画像エンコーダの検証実験用リポジトリです。
リアルタイム継続学習の文脈で画像エンコーダを学習し、その性能評価を行なっていきます。

## 実験環境のセットアップ

### 必要な環境

- x86_64のCPU
- Linux環境
- NVIDIA GPU（必須）

### セットアップ手順

1. **必要なソフトウェアのインストール**

   - git
   - make
   - docker

2. **実験環境の構築**

   ```bash
   # このリポジトリをクローン
   git clone https://github.com/MLShukai/pamiq-vision-exp.git
   cd pamiq-vision-exp

   # Dockerイメージをビルド（GPUサポート付き）
   make docker-build

   # Dockerコンテナを起動
   # LOG_DIRは実験ログを保存する場所を指定（デフォルトは./logs）
   make docker-up LOG_DIR=/path/to/log_dir

   # コンテナに接続
   make docker-attach
   ```

   VSCodeを使っている場合は、Container ToolsやDev Container拡張機能でアタッチすることも可能。

### 実験中のデバッグ

コンテナ内で開発する際の便利なコマンド：

```bash
# コードの自動フォーマット
make format

# 型チェック（エラーがないか確認）
make type

# テストを実行
make test

# 上記すべてを一度に実行
make run
```

## 依存関係の管理

```bash
# 新しいパッケージを追加
uv add <package_name>

# 開発用パッケージを追加
uv add --dev <package_name>

# 依存関係を最新に更新
uv sync

# 仮想環境を作り直す（問題が起きた時）
make venv
```
