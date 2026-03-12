# PAMIQ-Core 開発ガイド & チートシート

> 私が pamiq-core を使って開発する際のリファレンスドキュメント

元ドキュメント: https://mlshukai.github.io/pamiq-core/

## 📑 目次

01. [基本コンセプト](#%E5%9F%BA%E6%9C%AC%E3%82%B3%E3%83%B3%E3%82%BB%E3%83%97%E3%83%88)
02. [クイックスタート](#%E3%82%AF%E3%82%A4%E3%83%83%E3%82%AF%E3%82%B9%E3%82%BF%E3%83%BC%E3%83%88)
03. [コアコンポーネント](#%E3%82%B3%E3%82%A2%E3%82%B3%E3%83%B3%E3%83%9D%E3%83%BC%E3%83%8D%E3%83%B3%E3%83%88)
04. [データフロー](#%E3%83%87%E3%83%BC%E3%82%BF%E3%83%95%E3%83%AD%E3%83%BC)
05. [PyTorch統合](#pytorch%E7%B5%B1%E5%90%88)
06. [Gymnasium統合](#gymnasium%E7%B5%B1%E5%90%88)
07. [State Persistence](#state-persistence)
08. [Console制御](#console%E5%88%B6%E5%BE%A1)
09. [Wrappers](#wrappers)
10. [よくあるパターン](#%E3%82%88%E3%81%8F%E3%81%82%E3%82%8B%E3%83%91%E3%82%BF%E3%83%BC%E3%83%B3)
11. [トラブルシューティング](#%E3%83%88%E3%83%A9%E3%83%96%E3%83%AB%E3%82%B7%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0)

______________________________________________________________________

## 基本コンセプト

### PAMIQ-Core とは

**pamiq-core** は、推論と訓練の非同期実行を可能にする、最小限の機械学習フレームワークです。

**設計哲学**:

- **シンプルさ** - 直感的で使いやすいAPI
- **軽量** - 最小限の依存関係、最大のパフォーマンス
- **完全なスレッド抽象化** - 複雑なスレッド処理は内部で処理、外部にはシンプルなインターフェース

### アーキテクチャの全体像

```
┌─────────────────────────────────────────────────┐
│           Control Thread (main)                 │
│  - Web API Server (pamiq-console接続)           │
│  - State Persistence                            │
│  - System Lifecycle Management                  │
└─────────────────────────────────────────────────┘
           │                    │
    ┌──────▼──────┐      ┌─────▼──────┐
    │  Inference  │      │  Training  │
    │   Thread    │◄────►│   Thread   │
    └─────────────┘      └────────────┘
         │                     │
    Agent-Env            Model Training
    Interaction          & Sync
```

**3つの並行スレッド**:

1. **Control Thread** - システム全体の制御
2. **Inference Thread** - Agent-Environment のインタラクション実行
3. **Training Thread** - モデルの訓練とパラメータ同期

______________________________________________________________________

## クイックスタート

### インストール

```bash
# 基本インストール
pip install pamiq-core

# PyTorch統合込み
pip install pamiq-core[torch]

# Gymnasium統合込み
pip install pamiq-core[gym]

# キーボードコントローラー込み
pip install pamiq-core[kbctl]
```

### 最小限の実装例

```python
from pamiq_core import Agent, Environment, Interaction, LaunchConfig, launch
from typing import override

# 1. Agent を実装
class MyAgent(Agent[ObsType, ActionType]):
    @override
    def step(self, observation: ObsType) -> ActionType:
        # 観測から行動を決定
        return action

# 2. Environment を実装
class MyEnvironment(Environment[ObsType, ActionType]):
    @override
    def observe(self) -> ObsType:
        # 環境から観測を取得
        return observation

    @override
    def affect(self, action: ActionType) -> None:
        # 行動を環境に適用
        pass

# 3. システムを起動
launch(
    interaction=Interaction(MyAgent(), MyEnvironment()),
    models={},
    buffers={},
    trainers={},
    config=LaunchConfig(
        states_dir="./states",
        web_api_address=("localhost", 8391),
    )
)
```

______________________________________________________________________

## コアコンポーネント

### 1. Agent（エージェント）

エージェントは環境から観測を受け取り、行動を決定します。

#### 基本実装

```python
from pamiq_core import Agent
from typing import override

class MyAgent(Agent[ObservationType, ActionType]):
    @override
    def step(self, observation: ObservationType) -> ActionType:
        """観測を処理して行動を返す"""
        # 決定ロジックを実装
        return action
```

#### Inference Model の使用

```python
class MyAgent(Agent[ObservationType, ActionType]):
    @override
    def on_inference_models_attached(self) -> None:
        """推論モデルがアタッチされた時に呼ばれる"""
        self.policy = self.get_inference_model("policy")
        self.value = self.get_inference_model("value")

    @override
    def step(self, observation: ObservationType) -> ActionType:
        action = self.policy(observation)  # モデルで推論
        return action
```

#### Data Collector の使用

```python
class MyAgent(Agent[ObservationType, ActionType]):
    @override
    def on_data_collectors_attached(self) -> None:
        """データコレクターがアタッチされた時に呼ばれる"""
        self.collector = self.get_data_collector("experience")

    @override
    def step(self, observation: ObservationType) -> ActionType:
        action = self.decide_action(observation)

        # データを収集
        self.collector.collect({
            "observation": observation,
            "action": action,
            "timestamp": time.time()
        })

        return action
```

#### 階層的なエージェント（Composite Agent）

```python
class MainAgent(Agent[..., ...]):
    def __init__(self) -> None:
        # 子エージェントを作成
        self.navigation = NavigationAgent()
        self.perception = PerceptionAgent()

        # 親エージェントとして初期化（子エージェントに自動的にモデルとコレクターが伝播）
        super().__init__(agents={
            "navigation": self.navigation,
            "perception": self.perception
        })
```

### 2. Environment（環境）

環境は観測を生成し、エージェントの行動を受け取ります。

#### 基本実装

```python
from pamiq_core import Environment
from typing import override

class MyEnvironment(Environment[ObsType, ActionType]):
    @override
    def observe(self) -> ObsType:
        """現在の観測を返す"""
        return self.get_current_state()

    @override
    def affect(self, action: ActionType) -> None:
        """行動を環境に適用"""
        self.apply_action(action)
```

#### Modular Environment（センサー/アクチュエーター分離）

```python
from pamiq_core import ModularEnvironment, Sensor, Actuator
from typing import override

class CameraSensor(Sensor[np.ndarray]):
    @override
    def read(self) -> np.ndarray:
        """カメラから画像を取得"""
        return capture_image()

class MotorActuator(Actuator[dict]):
    @override
    def operate(self, action: dict) -> None:
        """モーターを制御"""
        control_motors(action)

# 組み合わせて使用
env = ModularEnvironment(CameraSensor(), MotorActuator())
```

#### 複数センサー/アクチュエーター

```python
from pamiq_core import SensorsDict, ActuatorsDict

sensors = SensorsDict({
    "camera": CameraSensor(),
    "audio": AudioSensor(),
    "imu": IMUSensor()
})

actuators = ActuatorsDict({
    "motor": MotorActuator(),
    "speaker": SpeakerActuator()
})

env = ModularEnvironment(sensors, actuators)
# agent.step() は dict[str, Any] を受け取り、dict[str, Any] を返す
```

### 3. Interaction（インタラクション）

エージェントと環境の相互作用を管理します。

#### 基本的な Interaction

```python
from pamiq_core import Interaction

interaction = Interaction(agent, environment)
```

#### Fixed Interval Interaction（固定間隔実行）

```python
from pamiq_core import FixedIntervalInteraction

# 10Hz（0.1秒間隔）で実行
interaction = FixedIntervalInteraction.with_sleep_adjustor(
    agent=agent,
    environment=environment,
    interval=0.1  # 秒単位
)
```

### 4. Model（モデル）

PAMIQ-Core は推論モデルと訓練モデルを分離します。

#### カスタムモデルの実装

```python
from pamiq_core import InferenceModel, TrainingModel
from typing import override

# 推論モデル
class MyInferenceModel(InferenceModel):
    def __init__(self, weights: list[float]):
        self.weights = weights

    @override
    def infer(self, features: list[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features))

# 訓練モデル
class MyTrainingModel(TrainingModel[MyInferenceModel]):
    def __init__(self):
        super().__init__(
            has_inference_model=True,      # 推論モデルを作成するか
            inference_thread_only=False    # 推論スレッドのみで使用するか
        )
        self.weights = [0.5, 0.3, -0.2]

    @override
    def _create_inference_model(self) -> MyInferenceModel:
        """推論モデルのインスタンスを作成"""
        return MyInferenceModel(self.weights.copy())

    @override
    def forward(self, features: list[float]) -> float:
        """訓練時のフォワードパス"""
        return sum(w * x for w, x in zip(self.weights, features))

    @override
    def sync_impl(self, inference_model: MyInferenceModel) -> None:
        """訓練モデルから推論モデルへパラメータを同期"""
        inference_model.weights = self.weights.copy()
```

**重要な設定パラメータ**:

- `has_inference_model=True`: 推論モデルを作成・管理する
- `has_inference_model=False`: 訓練のみに使用（推論モデルなし）
- `inference_thread_only=True`: 推論スレッドのみで使用（訓練で更新しない）
- `inference_thread_only=False`: 推論と訓練の両方で使用

### 5. Trainer（トレーナー）

モデルの訓練ロジックを実装します。

#### 基本実装

```python
from pamiq_core import Trainer
from typing import override

class MyTrainer(Trainer):
    def __init__(self):
        super().__init__(
            training_condition_data_user="experience",  # 訓練条件に使うバッファ
            min_buffer_size=1000,                       # 最小バッファサイズ
            min_new_data_count=100                      # 最小新規データ数
        )

    @override
    def on_training_models_attached(self) -> None:
        """訓練モデルがアタッチされた時に呼ばれる"""
        self.model = self.get_training_model("policy")

    @override
    def on_data_users_attached(self) -> None:
        """データユーザーがアタッチされた時に呼ばれる"""
        self.data = self.get_data_user("experience")

    @override
    def train(self) -> None:
        """訓練ロジックを実装"""
        self.data.update()  # データを更新
        experiences = self.data.get_data()

        # 訓練処理
        for batch in create_batches(experiences):
            loss = compute_loss(self.model, batch)
            optimize(loss)
```

#### カスタム訓練条件

```python
@override
def is_trainable(self) -> bool:
    """訓練を実行するかを判定"""
    if not super().is_trainable():
        return False

    # カスタム条件を追加
    return self.custom_condition_met()
```

### 6. Data（データ）

データの収集、保存、管理を行います。

#### 組み込みバッファ

```python
from pamiq_core.data.impls import (
    SequentialBuffer,           # 順次バッファ
    RandomReplacementBuffer     # ランダム置換バッファ
)

# 最大1024サンプルのランダム置換バッファ
buffer = RandomReplacementBuffer(max_size=1024)

# 最大512サンプルの順次バッファ
buffer = SequentialBuffer(max_size=512)
```

#### カスタムバッファの実装

```python
from pamiq_core.data import DataBuffer
from typing import override

class MyBuffer[T](DataBuffer[T, list[T]]):
    @override
    def __init__(self, max_size: int) -> None:
        super().__init__(max_queue_size=max_size)
        self._data: list[T] = []
        self._max_size = max_size

    @override
    def add(self, data: T) -> None:
        """データを追加"""
        if len(self._data) >= self._max_size:
            self._data.pop(0)
        self._data.append(data)

    @override
    def get_data(self) -> list[T]:
        """全データを取得"""
        return self._data.copy()

    @override
    def __len__(self) -> int:
        return len(self._data)
```

### 7. Launch（起動）

システム全体を起動します。

#### 基本的な起動

```python
from pamiq_core import launch, LaunchConfig

launch(
    interaction=interaction,
    models={"model_name": training_model},
    buffers={"buffer_name": data_buffer},
    trainers={"trainer_name": trainer},
    config=LaunchConfig(
        states_dir="./states",
        web_api_address=("localhost", 8391)
    )
)
```

#### LaunchConfig のオプション

```python
from pamiq_core.state_persistence import PeriodicSaveCondition, LatestStatesKeeper

LaunchConfig(
    # 状態の保存先
    states_dir="./states",

    # 既存の状態から再開
    saved_state_path="./states/checkpoint_001.state",

    # Web API（pamiq-console接続用）
    web_api_address=("localhost", 8391),  # Noneで無効化

    # 実行時間制限（秒）
    max_uptime=3600.0,  # 1時間で終了

    # 時間スケール（高速化・低速化）
    time_scale=2.0,  # 2倍速

    # 状態保存の条件
    save_state_condition=PeriodicSaveCondition(600.0),  # 10分毎

    # 状態の保持数
    states_keeper=LatestStatesKeeper(
        states_dir="./states",
        max_keep=5  # 最新5つのみ保持
    )
)
```

______________________________________________________________________

## データフロー

### Agent → Buffer → Trainer の完全な流れ

```python
# 1. データ収集（Agent - Inference Thread）
class MyAgent(Agent[obs, act]):
    def on_data_collectors_attached(self) -> None:
        self.collector = self.get_data_collector("experience")

    def step(self, obs: obs) -> act:
        action = self.decide(obs)
        self.collector.collect({"obs": obs, "act": action})  # 収集
        return action

# 2. データバッファ
buffer = RandomReplacementBuffer(max_size=10000)

# 3. データ使用（Trainer - Training Thread）
class MyTrainer(Trainer):
    def on_data_users_attached(self) -> None:
        self.data_user = self.get_data_user("experience")

    def train(self) -> None:
        self.data_user.update()  # Collector → Buffer へ転送
        data = self.data_user.get_data()  # Buffer から取得

        # 訓練処理
        for batch in create_batches(data):
            train_step(batch)

# 4. システムに登録
launch(
    interaction=Interaction(MyAgent(), env),
    models={"model": training_model},
    buffers={"experience": buffer},  # バッファを登録
    trainers={"trainer": MyTrainer()},
    config=config
)
```

______________________________________________________________________

## PyTorch統合

### TorchTrainingModel の使用

```python
import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel

# PyTorchモデルを定義
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

    def infer(self, x):
        """推論時の処理（異なる場合のみ定義）"""
        with torch.no_grad():
            return self.forward(x)

# PAMIQ-Core で使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TorchTrainingModel(
    PolicyNet(input_dim=10, output_dim=4),
    inference_procedure=PolicyNet.infer,  # 推論時の処理を指定
    device=device
)
```

### TorchTrainer の実装

```python
from pamiq_core.torch import TorchTrainer, OptimizersSetup
import torch.optim as optim

class MyTorchTrainer(TorchTrainer):
    def __init__(self):
        super().__init__(
            training_condition_data_user="experience",
            min_buffer_size=1000,
            min_new_data_count=100
        )

    @override
    def on_training_models_attached(self) -> None:
        super().on_training_models_attached()
        # TorchTrainingModelを取得（型チェック付き）
        self.policy = self.get_torch_training_model("policy", PolicyNet)

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """オプティマイザーを作成"""
        self.optimizer = optim.Adam(self.policy.model.parameters(), lr=1e-3)
        return {"optimizer": self.optimizer}

    @override
    def train(self) -> None:
        """訓練ロジック"""
        self.data_user.update()
        data = self.data_user.get_data()

        # バッチ作成
        dataloader = create_dataloader(data, batch_size=32)

        for batch in dataloader:
            # Forward
            output = self.policy(batch["input"])
            loss = compute_loss(output, batch["target"])

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

### 完全なPyTorchの例（VAE）

```python
# モデル定義
class Encoder(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        # ... エンコーダーの定義 ...

    def forward(self, x):
        # 訓練時の処理
        return distribution

    def infer(self, x):
        # 推論時の処理（サンプリングなし）
        with torch.no_grad():
            return self.forward(x).mean

class Decoder(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        # ... デコーダーの定義 ...

# システム構築
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "encoder": TorchTrainingModel(
        Encoder(feature_size=64),
        inference_procedure=Encoder.infer,
        device=device
    ),
    "decoder": TorchTrainingModel(
        Decoder(feature_size=64),
        has_inference_model=False,  # デコーダーは訓練のみ
        device=device
    )
}

launch(
    interaction=FixedIntervalInteraction.with_sleep_adjustor(
        agent=EncodingAgent(),
        environment=EncodingCheckEnv(feature_size=64),
        interval=0.1
    ),
    models=models,
    buffers={"observation": RandomReplacementBuffer(max_size=1024)},
    trainers={"vae": VAETrainer(max_epochs=3, batch_size=32)},
    config=LaunchConfig(states_dir="./states")
)
```

______________________________________________________________________

## Gymnasium統合

Gymnasium環境をPAMIQ-Coreで使用できます。

### GymEnvironment の使用

```python
from pamiq_core.gym import GymEnvironment

# 環境IDから作成
env = GymEnvironment("CartPole-v1")

# 既存のGymnasium環境を使用
import gymnasium as gym
gym_env = gym.make("CartPole-v1", render_mode="human")
env = GymEnvironment(gym_env)
```

### GymAgent の実装

```python
from pamiq_core.gym import GymAgent
import numpy as np
from typing import override

class CartPoleAgent(GymAgent[np.ndarray, int]):
    @override
    def on_reset(self, obs, info):
        """環境リセット時に呼ばれる"""
        return 0  # 初期行動

    @override
    def on_step(self, obs, reward, terminated, truncated, info):
        """各ステップで呼ばれる"""
        # シンプルなポリシー: ポールが右に傾いたら右に移動
        return 1 if obs[2] > 0 else 0
```

### 手動リセット

```python
class EarlyStoppingAgent(GymAgent[np.ndarray, int]):
    @override
    def on_step(self, obs, reward, terminated, truncated, info):
        # 条件を満たしたらリセットを要求
        if abs(obs[2]) > 0.5:
            self.need_reset = True
        return 1 if obs[2] > 0 else 0
```

### 完全なGymnasiumの例

```python
from pamiq_core import launch, LaunchConfig, Interaction
from pamiq_core.gym import GymEnvironment, GymAgent

class MyGymAgent(GymAgent[np.ndarray, int]):
    def on_inference_models_attached(self) -> None:
        self.policy = self.get_inference_model("policy")

    def on_data_collectors_attached(self) -> None:
        self.collector = self.get_data_collector("experience")

    def on_reset(self, obs, info):
        return self.policy(obs)

    def on_step(self, obs, reward, terminated, truncated, info):
        action = self.policy(obs)

        # 経験を収集
        self.collector.collect({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": terminated or truncated
        })

        return action

# システム起動
launch(
    interaction=Interaction(
        MyGymAgent(),
        GymEnvironment("CartPole-v1")
    ),
    models={"policy": policy_model},
    buffers={"experience": buffer},
    trainers={"trainer": trainer},
    config=LaunchConfig(states_dir="./states")
)
```

______________________________________________________________________

## State Persistence

### カスタム状態の保存と復元

```python
from pathlib import Path
from pamiq_core import Agent
from typing import override

class MyCustomAgent(Agent[float, int]):
    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.total_reward = 0.0
        self.learning_rate = 0.01

    @override
    def save_state(self, path: Path) -> None:
        """カスタム状態を保存"""
        super().save_state(path)  # 親クラスのメソッドを必ず呼ぶ

        path.mkdir(exist_ok=True)
        with open(path / "stats.txt", "w") as f:
            f.write(f"episode_count: {self.episode_count}\n")
            f.write(f"total_reward: {self.total_reward}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")

    @override
    def load_state(self, path: Path) -> None:
        """カスタム状態を復元"""
        super().load_state(path)  # 親クラスのメソッドを必ず呼ぶ

        try:
            with open(path / "stats.txt", "r") as f:
                for line in f:
                    key, value = line.strip().split(": ")
                    if key == "episode_count":
                        self.episode_count = int(value)
                    elif key == "total_reward":
                        self.total_reward = float(value)
                    elif key == "learning_rate":
                        self.learning_rate = float(value)
        except FileNotFoundError:
            # デフォルト値を使用
            pass
```

### 状態の保存条件

```python
from pamiq_core.state_persistence import PeriodicSaveCondition, LatestStatesKeeper

# 定期的に保存
config = LaunchConfig(
    states_dir="./states",
    save_state_condition=PeriodicSaveCondition(300.0),  # 5分毎
    states_keeper=LatestStatesKeeper(
        states_dir="./states",
        max_keep=10  # 最新10個を保持
    )
)
```

### 状態ディレクトリ構造

```
[timestamp].state/
├── interaction/
│   ├── agent/          # エージェントの状態
│   └── environment/    # 環境の状態
├── models/             # モデルの状態
├── data/               # データバッファの状態
├── trainers/           # トレーナーの状態
└── time.pkl            # 時間制御の状態
```

______________________________________________________________________

## Console制御

### pamiq-console の使用

```bash
# ローカルシステムに接続
pamiq-console --host localhost --port 8391

# リモートシステムに接続
pamiq-console --host 192.168.1.100 --port 8391
```

### 利用可能なコマンド

```
pamiq-console (active) > help

Available commands:
  h, help     - Show this help message
  p, pause    - Pause the system
  r, resume   - Resume the system
  save        - Save current state
  shutdown    - Shutdown the system (requires confirmation)
  q, quit     - Exit console (system continues running)
```

### Web API の使用

```bash
# ステータス取得
curl http://localhost:8391/api/status

# 一時停止
curl -X POST http://localhost:8391/api/pause

# 再開
curl -X POST http://localhost:8391/api/resume

# 状態保存
curl -X POST http://localhost:8391/api/save-state

# シャットダウン
curl -X POST http://localhost:8391/api/shutdown
```

### キーボードショートカットコントローラー

```bash
# インストール
pip install pamiq-core[kbctl]

# Linux の場合、依存関係をインストール
sudo apt-get install libevdev-dev build-essential

# 起動
pamiq-kbctl --host localhost --port 8391
```

**デフォルトのショートカット**:

- Windows/Linux: `Alt+Shift+P` (一時停止), `Alt+Shift+R` (再開)
- macOS: `Option+Shift+P` (一時停止), `Option+Shift+R` (再開)

______________________________________________________________________

## Wrappers

環境やエージェントの動作を拡張するためのラッパーを作成できます。

### EnvironmentWrapper の使用

```python
from pamiq_core import EnvironmentWrapper
from typing import override

class LoggingEnvironmentWrapper(EnvironmentWrapper[ObsType, ActionType]):
    """観測と行動をロギングするラッパー"""

    @override
    def observe(self) -> ObsType:
        obs = super().observe()
        print(f"Observation: {obs}")
        return obs

    @override
    def affect(self, action: ActionType) -> None:
        print(f"Action: {action}")
        super().affect(action)

# 使用
wrapped_env = LoggingEnvironmentWrapper(original_env)
```

### LambdaWrapper の使用

```python
from pamiq_core import LambdaWrapper

# 観測を正規化するラッパー
normalize_obs = LambdaWrapper(
    lambda obs: (obs - obs.mean()) / (obs.std() + 1e-8)
)

wrapped_env = normalize_obs.wrap_environment(original_env)
```

### SensorWrapper と ActuatorWrapper

```python
from pamiq_core import SensorWrapper, ActuatorWrapper

# センサーにノイズを追加
class NoisySensorWrapper(SensorWrapper[np.ndarray]):
    @override
    def read(self) -> np.ndarray:
        data = super().read()
        noise = np.random.randn(*data.shape) * 0.1
        return data + noise

# アクチュエーターに遅延を追加
class DelayedActuatorWrapper(ActuatorWrapper[ActionType]):
    def __init__(self, actuator, delay: float):
        super().__init__(actuator)
        self.delay = delay

    @override
    def operate(self, action: ActionType) -> None:
        time.sleep(self.delay)
        super().operate(action)
```

______________________________________________________________________

## よくあるパターン

### パターン1: 強化学習エージェント

```python
class RLAgent(Agent[np.ndarray, int]):
    def on_inference_models_attached(self) -> None:
        self.policy = self.get_inference_model("policy")
        self.value = self.get_inference_model("value")

    def on_data_collectors_attached(self) -> None:
        self.experience = self.get_data_collector("experience")

    def step(self, observation: np.ndarray) -> int:
        # ポリシーで行動を選択
        action_probs = self.policy(observation)
        action = np.random.choice(len(action_probs), p=action_probs)

        # 価値推定
        value = self.value(observation)

        # 経験を保存
        self.experience.collect({
            "state": observation,
            "action": action,
            "value": value
        })

        return action
```

### パターン2: 自己教師あり学習

```python
class SelfSupervisedAgent(Agent[torch.Tensor, torch.Tensor]):
    def on_inference_models_attached(self) -> None:
        self.encoder = self.get_inference_model("encoder")

    def on_data_collectors_attached(self) -> None:
        self.data = self.get_data_collector("observations")

    def step(self, observation: torch.Tensor) -> torch.Tensor:
        # エンコード
        encoding = self.encoder(observation)

        # 観測データを保存（訓練用）
        self.data.collect({"obs": observation.cpu()})

        # エンコーディングを返す（環境で検証など）
        return encoding
```

### パターン3: 状態の保存と復元

```python
from pathlib import Path
from pamiq_core.state_persistence import PeriodicSaveCondition, LatestStatesKeeper

# 初回実行
launch(
    interaction=interaction,
    models=models,
    buffers=buffers,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./states",
        save_state_condition=PeriodicSaveCondition(600.0),  # 10分毎保存
        states_keeper=LatestStatesKeeper(
            states_dir="./states",
            max_keep=5
        )
    )
)

# 再開
latest = sorted(Path("./states").glob("*.state"))[-1]
launch(
    interaction=interaction,
    models=models,
    buffers=buffers,
    trainers=trainers,
    config=LaunchConfig(
        states_dir="./states",
        saved_state_path=latest  # 最新の状態から再開
    )
)
```

### パターン4: マルチモーダル学習

```python
from pamiq_core import ModularEnvironment, SensorsDict, ActuatorsDict

# マルチモーダルセンサー
sensors = SensorsDict({
    "camera": CameraSensor(),
    "lidar": LidarSensor(),
    "audio": AudioSensor()
})

# マルチアクチュエーター
actuators = ActuatorsDict({
    "motor": MotorActuator(),
    "arm": ArmActuator(),
    "speaker": SpeakerActuator()
})

# エージェント
class MultimodalAgent(Agent[dict, dict]):
    def on_inference_models_attached(self) -> None:
        self.vision_encoder = self.get_inference_model("vision")
        self.audio_encoder = self.get_inference_model("audio")
        self.policy = self.get_inference_model("policy")

    def step(self, observation: dict) -> dict:
        # 各モダリティをエンコード
        vision_feat = self.vision_encoder(observation["camera"])
        audio_feat = self.audio_encoder(observation["audio"])

        # 統合して行動を決定
        combined = torch.cat([vision_feat, audio_feat], dim=-1)
        actions = self.policy(combined)

        return {
            "motor": actions[:3],
            "arm": actions[3:6],
            "speaker": actions[6:]
        }

# 起動
launch(
    interaction=Interaction(
        MultimodalAgent(),
        ModularEnvironment(sensors, actuators)
    ),
    models=models,
    buffers=buffers,
    trainers=trainers,
    config=config
)
```

### パターン5: リモートコンソール制御

```bash
# システム起動時に web_api_address を設定
# config=LaunchConfig(web_api_address=("localhost", 8391))

# 別ターミナルから接続
pamiq-console --host localhost --port 8391

# コンソールでコマンド実行
pamiq-console (active) > pause     # 一時停止
pamiq-console (paused) > resume    # 再開
pamiq-console (active) > save      # 状態保存
pamiq-console (active) > shutdown  # シャットダウン
```

______________________________________________________________________

## トラブルシューティング

### Q: モデルのパラメータが更新されない

**A**: `sync_impl()` が正しく実装されているか確認してください。

```python
@override
def sync_impl(self, inference_model: MyInferenceModel) -> None:
    # パラメータを確実にコピー
    inference_model.weights = self.weights.copy()  # ✓
    # inference_model.weights = self.weights  # ✗（参照コピー）
```

### Q: データが収集されない

**A**: `DataUser.update()` を呼び出していますか？

```python
def train(self) -> None:
    self.data_user.update()  # これを忘れずに！
    data = self.data_user.get_data()
```

### Q: 訓練が実行されない

**A**: 訓練条件を確認してください。

```python
# min_buffer_size と min_new_data_count を確認
trainer = MyTrainer(
    training_condition_data_user="experience",
    min_buffer_size=1000,      # バッファに1000サンプル必要
    min_new_data_count=100     # 100個の新規データが必要
)
```

### Q: GPU が使用されない（PyTorch）

**A**: デバイスを明示的に指定してください。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TorchTrainingModel(net, device=device)
```

### Q: 時間制御が効かない

**A**: `FixedIntervalInteraction` を使用していますか？

```python
# ✓ 固定間隔で実行
interaction = FixedIntervalInteraction.with_sleep_adjustor(
    agent, environment, interval=0.1
)

# ✗ 通常のInteractionは間隔制御なし
interaction = Interaction(agent, environment)
```

### Q: pamiq-console が接続できない

**A**: Web API が有効になっているか確認してください。

```python
# ✓ Web APIを有効化
config = LaunchConfig(web_api_address=("localhost", 8391))

# ✗ Web APIが無効
config = LaunchConfig(web_api_address=None)
```

### Q: 状態が保存されない

**A**: `save_state_condition` が設定されているか確認してください。

```python
from pamiq_core.state_persistence import PeriodicSaveCondition

config = LaunchConfig(
    states_dir="./states",
    save_state_condition=PeriodicSaveCondition(600.0)  # 10分毎
)
```

______________________________________________________________________

## 参考リソース

### 公式ドキュメント

- [PAMIQ-Core Documentation](https://mlshukai.github.io/pamiq-core/)
- [GitHub - PAMIQ-Core](https://github.com/MLShukai/pamiq-core)

### サンプル

- [minimum.py](pamiq-core/samples/minimum.py) - 最小限の実装
- [vae-torch/](pamiq-core/samples/vae-torch/) - VAEの訓練例

### 関連プロジェクト

- [pamiq-recorder](https://github.com/MLShukai/pamiq-recorder) - 記録ライブラリ
- [pamiq-io](https://github.com/MLShukai/pamiq-io) - I/Oライブラリ
- [pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat) - VRChat統合

______________________________________________________________________

## チートシート - よく使うコマンド

### インポート

```python
# コア
from pamiq_core import (
    Agent, Environment, Interaction,
    FixedIntervalInteraction, ModularEnvironment,
    Sensor, Actuator, SensorsDict, ActuatorsDict,
    InferenceModel, TrainingModel, Trainer,
    DataBuffer, DataCollector, DataUser,
    launch, LaunchConfig
)

# PyTorch
from pamiq_core.torch import TorchTrainingModel, TorchTrainer, OptimizersSetup

# データバッファ
from pamiq_core.data.impls import (
    SequentialBuffer, RandomReplacementBuffer
)

# Gymnasium
from pamiq_core.gym import GymEnvironment, GymAgent

# State Persistence
from pamiq_core.state_persistence import (
    PeriodicSaveCondition, LatestStatesKeeper
)

# Wrappers
from pamiq_core import (
    EnvironmentWrapper, LambdaWrapper,
    SensorWrapper, ActuatorWrapper
)
```

### 基本的なフロー

```python
# 1. コンポーネント作成
agent = MyAgent()
environment = MyEnvironment()
interaction = Interaction(agent, environment)

# 2. モデル・バッファ・トレーナー
models = {"model_name": TorchTrainingModel(net, device=device)}
buffers = {"buffer_name": RandomReplacementBuffer(max_size=10000)}
trainers = {"trainer_name": MyTrainer()}

# 3. 起動
launch(
    interaction,
    models,
    buffers,
    trainers,
    LaunchConfig(
        states_dir="./states",
        web_api_address=("localhost", 8391)
    )
)
```

### よく使うコマンド

```bash
# pamiq-console
pamiq-console --host localhost --port 8391

# キーボードコントローラー
pamiq-kbctl --host localhost --port 8391

# API（curl）
curl http://localhost:8391/api/status
curl -X POST http://localhost:8391/api/pause
curl -X POST http://localhost:8391/api/resume
curl -X POST http://localhost:8391/api/save-state
```

______________________________________________________________________

**最終更新**: 2026-01-11
**バージョン**: pamiq-core 準拠
