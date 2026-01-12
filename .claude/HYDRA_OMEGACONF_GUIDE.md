# Hydra & OmegaConf 完全ガイド

## 目次

01. [基本概念](#%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5)
02. [プロジェクト構造](#%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E6%A7%8B%E9%80%A0)
03. [設定ファイルの書き方](#%E8%A8%AD%E5%AE%9A%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%81%AE%E6%9B%B8%E3%81%8D%E6%96%B9)
04. [Defaults機構](#defaults%E6%A9%9F%E6%A7%8B)
05. [変数参照と補間](#%E5%A4%89%E6%95%B0%E5%8F%82%E7%85%A7%E3%81%A8%E8%A3%9C%E9%96%93)
06. [カスタムリゾルバ](#%E3%82%AB%E3%82%B9%E3%82%BF%E3%83%A0%E3%83%AA%E3%82%BE%E3%83%AB%E3%83%90)
07. [hydra.utils.instantiate](#hydra-utils-instantiate)
08. [CLIでのオーバーライド](#cli%E3%81%A7%E3%81%AE%E3%82%AA%E3%83%BC%E3%83%90%E3%83%BC%E3%83%A9%E3%82%A4%E3%83%89)
09. [実践パターン](#%E5%AE%9F%E8%B7%B5%E3%83%91%E3%82%BF%E3%83%BC%E3%83%B3)
10. [デバッグとトラブルシューティング](#%E3%83%87%E3%83%90%E3%83%83%E3%82%B0%E3%81%A8%E3%83%88%E3%83%A9%E3%83%96%E3%83%AB%E3%82%B7%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0)

______________________________________________________________________

## 基本概念

### Hydraとは

- Pythonアプリケーションの設定を階層的に管理するフレームワーク
- YAMLファイルで設定を記述し、コマンドラインから動的にオーバーライド可能
- 複数の設定を組み合わせて柔軟に実験設定を管理

### OmegaConfとは

- Hydraが内部で使用する設定管理ライブラリ
- YAML/辞書/dataclassを統一的に扱う
- 変数補間や型チェックをサポート

### 基本的な使い方

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # cfg.key でアクセス
    print(cfg.experiment_name)

    # YAMLとして表示
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
```

**パラメータ説明:**

- `config_path`: 設定ファイルのディレクトリ（相対パスまたは絶対パス）
- `config_name`: メインの設定ファイル名（拡張子なし）
- `version_base`: Hydraのバージョン指定（"1.3"推奨）

______________________________________________________________________

## プロジェクト構造

### 推奨ディレクトリ構成

```
project/
├── src/
│   ├── configs/
│   │   ├── train.yaml              # メイン設定
│   │   ├── interaction/
│   │   │   └── default.yaml
│   │   ├── models/
│   │   │   ├── stacked_hiddens.yaml
│   │   │   └── hierarchical.yaml
│   │   ├── trainers/
│   │   │   └── default.yaml
│   │   ├── buffers/
│   │   │   └── default.yaml
│   │   ├── experiment/
│   │   │   └── hierarchical.yaml   # 実験設定
│   │   ├── shared/
│   │   │   └── default.yaml        # 共通設定
│   │   ├── paths/
│   │   │   └── default.yaml
│   │   ├── launch/
│   │   │   └── default.yaml
│   │   └── hydra/
│   │       └── default.yaml        # Hydra自体の設定
│   ├── train.py                    # エントリーポイント
│   └── exp/
│       ├── oc_resolvers.py         # カスタムリゾルバ
│       └── instantiations.py       # インスタンス化ロジック
└── pyproject.toml
```

**設計原則:**

- **カテゴリ別フォルダ**: models, trainers, buffersなどコンポーネントごとに分離
- **experiment/**: 実験固有の設定をまとめる（複数の設定をオーバーライド）
- **shared/**: 全体で共有する設定値（device, dtypeなど）
- **hydra/**: ログ出力、ワーキングディレクトリなどHydra自体の動作設定

______________________________________________________________________

## 設定ファイルの書き方

### 基本構文

```yaml
# 単純な値
experiment_name: "my_experiment"
log_level: INFO
batch_size: 32

# ネストした値
model:
  hidden_dim: 256
  num_layers: 4
  dropout: 0.1

# リスト
tags:
  - experiment
  - baseline
```

### `_target_` によるクラス指定

Hydraでオブジェクトをインスタンス化するための特殊キー:

```yaml
interaction:
  _target_: pamiq_core.FixedIntervalInteraction.with_sleep_adjustor
  interval: 0.1

  agent:
    _target_: exp.agents.integration.IntegratedCuriosityFramework
    param1: value1
```

**重要:**

- `_target_`: インスタンス化するクラス/関数の完全修飾パス
- その他のキーは `__init__` の引数として渡される
- `hydra.utils.instantiate(cfg.interaction)` で実体化

### ネストしたインスタンス化

```yaml
policy_value:
  _target_: pamiq_core.torch.TorchTrainingModel
  device: cuda:0
  dtype: float32
  model:
    _target_: exp.models.policy.StackedHiddenPiV
    dim: 1024
    core_model:
      _target_: exp.models.components.qlstm.QLSTM
      dim: 1024
      depth: 8
```

`hydra.utils.instantiate()` は再帰的に動作し、ネストされた `_target_` も自動でインスタンス化される。

______________________________________________________________________

## Defaults機構

### 基本的なdefaults

`train.yaml`:

```yaml
# @package _global_

defaults:
  - _self_                          # この設定自身（順序重要）
  - interaction: default            # interaction/default.yaml を読み込む
  - models: stacked_hiddens         # models/stacked_hiddens.yaml
  - trainers: stacked_hiddens
  - buffers: stacked_hiddens
  - launch: default
  - paths: default
  - hydra: default
  - shared: default
  - experiment: null                # オプショナル

experiment_name: "default"
tags: []
```

**動作:**

1. `interaction/default.yaml` の内容が `cfg.interaction` にマージされる
2. `models/stacked_hiddens.yaml` の内容が `cfg.models` にマージされる
3. 各ファイルは独立した名前空間を持つ

### `@package _global_` とは

```yaml
# @package _global_

defaults:
  - override /interaction/agent: hierarchical
```

- **`@package _global_`**: この設定がグローバルスコープに配置されることを示す
- **なしの場合**: ファイル名に基づいた名前空間に配置される

### Override構文

実験設定ファイル（`experiment/hierarchical.yaml`）:

```yaml
# @package _global_

defaults:
  - override /interaction/agent: hierarchical
  - override /trainers: hierarchical
  - override /models: hierarchical
  - override /buffers: hierarchical

shared:
  num_hierarchical_layers: 2
  max_imagination_steps: 3

experiment_name: "hierarchical"
```

**ポイント:**

- `override /path/to/key: value` で既存のdefaultsを上書き
- `/` から始まるパスはグローバルルートを指す
- 実験ファイル一つで複数の設定を切り替え可能

### Defaults順序の重要性

```yaml
defaults:
  - shared: default
  - _self_              # この位置が重要
  - experiment: null

custom_value: 123
```

- `_self_` より前: 他の設定に上書きされる可能性
- `_self_` より後: 自分が他の設定を上書き
- **基本方針**: `_self_` を最初に配置し、後続の設定で上書きできるようにする

______________________________________________________________________

## 変数参照と補間

### 基本的な参照

```yaml
shared:
  device: cuda:0
  dtype: float32

model:
  device: ${shared.device}    # shared.deviceを参照
  dtype: ${shared.dtype}
```

### 相対参照

```yaml
policy_value:
  dim: 1024
  core_model:
    dim: ${..dim}             # 親の dim を参照（1024）
    hidden_dim: ${.dim}       # 自分の dim を参照
```

- `.` : 現在の階層
- `..` : 親の階層
- `...` : 祖父の階層

### クロスリファレンス

```yaml
models:
  policy_value:
    model:
      obs_info:
        dim: 128

  forward_dynamics:
    model:
      obs_info: ${models.policy_value.model.obs_info}  # 他モデルの設定を参照
```

**利点:**

- 設定の一貫性を保つ
- 変更が自動的に伝播

### リスト要素の参照

```yaml
shared:
  image:
    height: 144
    width: 144

model:
  image_size:
    - ${shared.image.height}
    - ${shared.image.width}
```

______________________________________________________________________

## カスタムリゾルバ

### リゾルバとは

`${resolver_name: argument}` の形式でYAML内で関数を呼び出す機能

### 標準リゾルバ

```yaml
# 環境変数
data_path: ${oc.env:HOME}/data

# 現在時刻
log_dir: ./logs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### カスタムリゾルバの登録

`exp/oc_resolvers.py`:

```python
import torch
from omegaconf import OmegaConf

_registered = False

def register_custom_resolvers() -> None:
    global _registered

    if not _registered:
        # Python eval
        OmegaConf.register_new_resolver("python.eval", eval)

        # PyTorch device
        OmegaConf.register_new_resolver("torch.device", torch.device)

        # PyTorch dtype
        OmegaConf.register_new_resolver("torch.dtype", convert_dtype_str_to_torch_dtype)

        _registered = True

def convert_dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """文字列 'float32' を torch.float32 に変換"""
    if hasattr(torch, dtype_str):
        dtype = getattr(torch, dtype_str)
    else:
        raise ValueError(f"Dtype name {dtype_str} does not exist!")

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Dtype name {dtype_str} is not dtype! Object: {dtype}")
    return dtype
```

### 使用例

```yaml
shared:
  device: cuda:0
  dtype: float32

model:
  # リゾルバで変換
  device: ${torch.device:${shared.device}}      # → torch.device('cuda:0')
  dtype: ${torch.dtype:${shared.dtype}}         # → torch.float32

  hidden_dim: 256
  ff_dim: ${python.eval:"${.hidden_dim} * 4"}   # → 1024
```

### train.pyでの初期化

```python
from exp.oc_resolvers import register_custom_resolvers

# Hydra decorator より前に登録
register_custom_resolvers()

@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # リゾルバが既に利用可能
    # ただし、torch.device などはインスタンス化前に解決が必要
    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"

    # これで instantiate 時に適切な型になる
    models = instantiate_models(cfg)
```

### よく使うカスタムリゾルバパターン

```python
# 算術演算
OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)

# 条件分岐
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

# パス結合
import os
OmegaConf.register_new_resolver("path.join", os.path.join)
```

使用:

```yaml
dim: 256
ff_dim: ${mul:${.dim},4}                    # 1024
use_gpu: true
device: ${if:${.use_gpu},"cuda:0","cpu"}
```

______________________________________________________________________

## hydra.utils.instantiate

### 基本的な使い方

```python
import hydra
from omegaconf import DictConfig

def instantiate_interaction(cfg: DictConfig):
    # cfg.interaction に _target_ が含まれている必要がある
    interaction = hydra.utils.instantiate(cfg.interaction)
    return interaction
```

### _target_ の種類

#### 1. クラスのインスタンス化

```yaml
interaction:
  _target_: pamiq_core.FixedIntervalInteraction.with_sleep_adjustor
  interval: 0.1
```

```python
# 以下と等価:
from pamiq_core import FixedIntervalInteraction
interaction = FixedIntervalInteraction.with_sleep_adjustor(interval=0.1)
```

#### 2. クラスメソッド/スタティックメソッド

```yaml
inference_procedure:
  _target_: hydra.utils.get_method
  path: exp.models.policy.StackedHiddenPiV.forward_with_no_len
```

```python
# get_method でメソッドオブジェクトを取得
method = hydra.utils.get_method("exp.models.policy.StackedHiddenPiV.forward_with_no_len")
```

#### 3. オブジェクトの取得

```yaml
action_choices:
  _target_: hydra.utils.get_object
  path: exp.envs.vrchat.OSC_ACTION_CHOICES
```

```python
# モジュールレベルの変数を取得
from exp.envs.vrchat import OSC_ACTION_CHOICES
```

### ネストしたインスタンス化

```python
def instantiate_models(cfg: DictConfig) -> dict[str, TorchTrainingModel]:
    models_dict = {}

    for name, model_cfg in cfg.models.items():
        logger.info(f"Instantiating model: '{name}'...")

        # ネストした _target_ も自動で処理される
        model = hydra.utils.instantiate(model_cfg)

        # リストが返ることもある
        if isinstance(model, list):
            for i, m in enumerate(model):
                models_dict[f"{name}{i}"] = m
        else:
            models_dict[name] = model

    return models_dict
```

### _partial_ による遅延インスタンス化

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true    # functools.partial として扱う
  lr: 0.001
  weight_decay: 0.01
```

```python
from functools import partial

optimizer_factory = hydra.utils.instantiate(cfg.optimizer)
# optimizer_factory は partial(AdamW, lr=0.001, weight_decay=0.01)

# 後でパラメータを渡す
optimizer = optimizer_factory(model.parameters())
```

### _convert_ による型変換制御

```yaml
model:
  _target_: MyModel
  _convert_: partial    # デフォルトは "partial"
  layers: [64, 128, 256]
```

- `none`: 変換なし（DictConfigのまま）
- `partial`: ListConfigをlist、DictConfigをdictに変換
- `all`: 完全に変換

______________________________________________________________________

## CLIでのオーバーライド

### 基本構文

```bash
python train.py key=value
```

### 単一値のオーバーライド

```bash
# 実験名を変更
python train.py experiment_name=test_run

# ログレベル変更
python train.py log_level=DEBUG

# ネストした値
python train.py shared.device=cuda:1
python train.py model.hidden_dim=512
```

### defaultsのオーバーライド

```bash
# models設定を切り替え
python train.py models=hierarchical

# 複数同時
python train.py models=hierarchical trainers=hierarchical

# experimentを指定（複数のdefaultsをまとめて切り替え）
python train.py experiment=hierarchical
```

### リストの操作

```bash
# リスト全体を置換
python train.py tags=[exp1,exp2,exp3]

# 追加
python train.py +tags=[new_tag]

# リスト要素の変更
python train.py 'tags[0]=first_tag'
```

### キーの追加・削除

```bash
# 新しいキーを追加
python train.py +new_key=value
python train.py +model.new_param=123

# キーを削除
python train.py ~tags
python train.py ~model.dropout
```

### 複雑な値の指定

```bash
# 辞書
python train.py 'model={hidden_dim: 512, dropout: 0.1}'

# リスト内辞書
python train.py 'layers=[{dim: 64}, {dim: 128}]'

# 引用符が必要な場合
python train.py 'experiment_name="My Experiment"'
```

### Multirun（複数設定で実行）

```bash
# hidden_dimを3パターンで実行
python train.py -m model.hidden_dim=256,512,1024

# 複数パラメータの組み合わせ
python train.py -m model.hidden_dim=256,512 model.dropout=0.1,0.2
# → 4つの組み合わせ（2x2）を実行

# defaultsとの組み合わせ
python train.py -m experiment=baseline,hierarchical shared.device=cuda:0,cuda:1
```

**出力ディレクトリ:**

- Multirunでは `multirun/YYYY-MM-DD/HH-MM-SS/0`, `1`, `2`... と番号付きで保存される

### コマンドライン例（実践）

```bash
# デフォルト実行
python train.py

# GPU変更して実行
python train.py shared.device=cuda:1

# 実験設定を切り替え
python train.py experiment=hierarchical

# バッチサイズとLRを変更
python train.py model.batch_size=64 +optimizer.lr=0.0001

# タグを追加
python train.py tags=[baseline,large_model] experiment_name=baseline_v2

# 複数GPUでハイパラサーチ
python train.py -m shared.device=cuda:0,cuda:1 \
  model.hidden_dim=256,512,1024 \
  experiment_name=hyperparam_search
```

______________________________________________________________________

## 実践パターン

### パターン1: 共通設定の管理

`shared/default.yaml`:

```yaml
image:
  height: 144
  width: 144
  channels: 3

max_imagination_steps: 1
device: cuda:0
dtype: float32
```

**使い方:**

```python
@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 全コンポーネントで共通の設定にアクセス
    device = cfg.shared.device
    dtype = cfg.shared.dtype
    image_shape = (cfg.shared.image.height, cfg.shared.image.width)
```

### パターン2: 実験設定のバージョン管理

`experiment/baseline.yaml`:

```yaml
# @package _global_

defaults:
  - override /models: stacked_hiddens
  - override /trainers: stacked_hiddens

experiment_name: "baseline"
tags: [baseline, stacked]

shared:
  max_imagination_steps: 1
```

`experiment/advanced.yaml`:

```yaml
# @package _global_

defaults:
  - override /models: hierarchical
  - override /trainers: hierarchical

experiment_name: "advanced_hierarchical"
tags: [advanced, hierarchical]

shared:
  max_imagination_steps: 3
  num_hierarchical_layers: 2
```

**切り替え:**

```bash
python train.py experiment=baseline
python train.py experiment=advanced
```

### パターン3: 条件付きデフォルト

`interaction/default.yaml`:

```yaml
defaults:
  - environment: vrchat
  - agent: adversarial

_target_: pamiq_core.FixedIntervalInteraction.with_sleep_adjustor
interval: 0.1

agent:
  _target_: exp.agents.integration.IntegratedCuriosityFramework
```

- `interaction/environment/vrchat.yaml` が自動的に読み込まれる
- `interaction/agent/adversarial.yaml` も読み込まれる
- 階層的な設定の組み合わせが可能

### パターン4: 設定の再利用

```yaml
models:
  policy_value:
    model:
      obs_info:
        _target_: exp.models.utils.ObsInfo
        dim: 128
        num_tokens: 144
        dim_hidden: ${..dim}

  forward_dynamics:
    model:
      # policy_valueと同じobs_infoを使う
      obs_info: ${models.policy_value.model.obs_info}

      action_info:
        _target_: exp.models.utils.ActionInfo
        # policy_valueのaction_choicesを参照
        choices: ${models.policy_value.model.action_choices}
        dim: ${..dim}
```

### パターン5: 動的な値計算

```yaml
model:
  dim: 1024
  # dimの4倍を計算
  dim_ff_hidden: ${python.eval:"${.dim} * 4"}  # → 4096

image:
  height: 144
  width: 144

num_patches:
  _target_: exp.models.jepa.compute_image_jepa_output_patch_count
  image_size:
    - ${shared.image.height}
    - ${shared.image.width}
  patch_size: 12
  output_downsample: 3
```

### パターン6: Hydraの出力ディレクトリ管理

`hydra/default.yaml`:

```yaml
run:
  dir: ${paths.log_dir}/${experiment_name}/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}

sweep:
  dir: ${paths.log_dir}/${experiment_name}/multiruns/${now:%Y-%m-%d}/${now:%H-%M-%S}
  subdir: ${hydra.job.num}
```

**効果:**

- 実験ごとにディレクトリが分かれる
- 日時でソート可能
- Multirun時は番号付き

### パターン7: ロギング設定

```yaml
job_logging:
  root:
    handlers: [root_console, root_file]
    level: ${log_level}    # train.yaml で定義されたlog_levelを参照

  handlers:
    root_console:
      class: logging.StreamHandler
      stream: ext://sys.stdout
      formatter: colorlog

    root_file:
      class: logging.handlers.TimedRotatingFileHandler
      filename: ${hydra.runtime.output_dir}/root.log
      when: D
      formatter: simple
```

______________________________________________________________________

## デバッグとトラブルシューティング

### 設定の確認

```python
@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # YAML形式で出力
    print(OmegaConf.to_yaml(cfg))

    # 特定キーのみ
    print(OmegaConf.to_yaml(cfg.models))

    # 補間を解決してから出力
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
```

### CLIでの設定確認（実行せずに）

```bash
# --cfg job でマージされた設定を表示
python train.py --cfg job

# --cfg hydra でHydra自身の設定を表示
python train.py --cfg hydra

# --help で利用可能なオプションを表示
python train.py --help
```

### よくあるエラー

#### 1. `InterpolationKeyError: Interpolation key 'xxx' not found`

**原因:** 参照先のキーが存在しない

```yaml
model:
  device: ${shared.device}  # shared.device が存在しない
```

**解決:**

- 参照パスが正しいか確認
- defaultsで必要な設定ファイルが読み込まれているか確認

#### 2. `ConfigAttributeError: Key 'xxx' is not in struct`

**原因:** 構造モードで未定義のキーにアクセス

```python
cfg.nonexistent_key = "value"  # エラー
```

**解決:**

```python
OmegaConf.set_struct(cfg, False)  # 構造モードを無効化
cfg.nonexistent_key = "value"     # OK
OmegaConf.set_struct(cfg, True)   # 再度有効化
```

#### 3. `InstantiationException: Error locating target 'xxx'`

**原因:** `_target_` で指定されたクラス/関数が見つからない

**解決:**

- import パスが正しいか確認
- モジュールがインストールされているか確認
- タイポがないか確認

#### 4. Defaults順序の問題

**症状:** 設定が期待通りにオーバーライドされない

```yaml
defaults:
  - shared: default
  - experiment: baseline

shared:
  device: cuda:0  # experiment/baseline.yaml の設定に上書きされてしまう
```

**解決:**

```yaml
defaults:
  - shared: default
  - experiment: baseline
  - _self_  # 自分の設定を最後に適用

shared:
  device: cuda:0  # これが優先される
```

### ログを活用したデバッグ

```python
import logging

logger = logging.getLogger(__name__)

def instantiate_models(cfg: DictConfig):
    logger.info("Instantiating Models...")
    logger.debug(f"Config: {OmegaConf.to_yaml(cfg.models)}")

    for name, model_cfg in cfg.models.items():
        logger.info(f"Instantiating model: '{name}'...")
        try:
            model = hydra.utils.instantiate(model_cfg)
        except Exception as e:
            logger.error(f"Failed to instantiate {name}: {e}")
            logger.debug(f"Config was: {OmegaConf.to_yaml(model_cfg)}")
            raise

    return models_dict
```

### 設定のバリデーション

```python
def validate_config(cfg: DictConfig) -> None:
    """設定の妥当性チェック"""
    assert cfg.shared.device in ["cpu", "cuda:0", "cuda:1"], \
        f"Invalid device: {cfg.shared.device}"

    assert cfg.shared.image.height > 0, "Image height must be positive"

    if "models" in cfg:
        for name in cfg.models:
            assert "_target_" in cfg.models[name], \
                f"Model {name} missing _target_"

@hydra.main("./configs", "train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    validate_config(cfg)
    # ...
```

______________________________________________________________________

## クイックリファレンス

### YAML構文

| 機能           | 構文                      | 例                          |
| -------------- | ------------------------- | --------------------------- |
| 変数参照       | `${path.to.key}`          | `device: ${shared.device}`  |
| 相対参照       | `${.key}`, `${..key}`     | `dim: ${..hidden_dim}`      |
| リゾルバ       | `${resolver:arg}`         | `${now:%Y-%m-%d}`           |
| インスタンス化 | `_target_: path.to.Class` | `_target_: torch.nn.Linear` |
| Partial        | `_partial_: true`         | functools.partialとして扱う |
| パッケージ     | `# @package _global_`     | グローバルスコープに配置    |

### CLI構文

| 操作           | 構文           | 例                          |
| -------------- | -------------- | --------------------------- |
| オーバーライド | `key=value`    | `model.dim=512`             |
| 追加           | `+key=value`   | `+new_param=123`            |
| 削除           | `~key`         | `~model.dropout`            |
| Defaults変更   | `group=option` | `models=hierarchical`       |
| Multirun       | `-m key=v1,v2` | `-m dim=256,512`            |
| 設定表示       | `--cfg job`    | `python train.py --cfg job` |

### Python API

```python
# 設定読み込み
@hydra.main(config_path="./configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig): ...

# インスタンス化
obj = hydra.utils.instantiate(cfg.model)

# メソッド取得
method = hydra.utils.get_method("module.Class.method")

# オブジェクト取得
obj = hydra.utils.get_object("module.CONSTANT")

# YAML出力
yaml_str = OmegaConf.to_yaml(cfg)

# 補間解決
OmegaConf.resolve(cfg)

# カスタムリゾルバ登録
OmegaConf.register_new_resolver("name", func)
```

______________________________________________________________________

## ベストプラクティス

1. **設定はカテゴリごとに分割**

   - 1ファイル1責務を心がける
   - 再利用可能な粒度で設計

2. **shared設定を活用**

   - 複数箇所で使う値はsharedに集約
   - デバイス、dtype、画像サイズなど

3. **experiment設定でバリエーション管理**

   - 実験ごとに1ファイル
   - 再現性のため、設定をバージョン管理

4. **変数参照で一貫性を保つ**

   - 同じ値を複数箇所に書かない
   - 変更時の修正漏れを防ぐ

5. **カスタムリゾルバは事前登録**

   - `@hydra.main` より前に `register_custom_resolvers()` を呼ぶ
   - グローバル変数で二重登録を防止

6. **設定の確認を習慣化**

   - `--cfg job` で実際の設定を確認
   - `OmegaConf.to_yaml()` でログに出力

7. **構造化ログで追跡**

   - Hydraの出力ディレクトリを活用
   - 実験名、タグで整理

8. **CLIオーバーライドを活用**

   - 手早く実験条件を変更
   - Multirunでハイパーパラメータサーチ

______________________________________________________________________

## まとめ

Hydra/OmegaConfは、機械学習実験の設定管理において非常に強力なツール。以下を押さえておけば、ほとんどのユースケースに対応できる:

1. **階層的な設定ファイル構造** (defaults機構)
2. **変数補間による設定の再利用** (${...})
3. **カスタムリゾルバによる動的な値生成**
4. **hydra.utils.instantiate による依存性注入**
5. **CLIからの柔軟なオーバーライド**

このガイドを参考に、自分のプロジェクトに合わせた設定管理システムを構築すること。
