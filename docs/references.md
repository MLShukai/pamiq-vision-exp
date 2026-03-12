# 参考文献

## Continuous / Continual Learning

| 論文                                                                                                                              | 年   | 備考                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------- | ---- | -------------------------------------------------------------------- |
| [Learning from One Continuous Video Stream](https://arxiv.org/abs/2312.00598) (Carreira et al., CVPR 2024)                        | 2024 | 単一連続動画からのオンライン学習。本プロジェクトの問題設定に最も近い |
| [Orthogonal Gradient Descent for Continual Learning](https://arxiv.org/abs/1910.07104) (Farajtabar et al., AISTATS 2020)          | 2020 | 勾配の直交射影による破壊的忘却の回避                                 |
| [On the Theory of Continual Learning with Gradient Descent for Neural Networks](https://arxiv.org/abs/2510.05573) (Taheri et al.) | 2025 | 勾配降下法による継続学習の理論的解析                                 |

## JEPA ファミリー

| 論文                                                                                                                                                       | 年   | 備考                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----------------------------------------------------------- |
| [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (Assran et al., CVPR 2023) | 2023 | 画像版 JEPA。データ拡張なしで意味的な表現を学習             |
| [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (Bardes et al.)                   | 2024 | 動画版 JEPA。本プロジェクトの主要手法                       |
| [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) (Bardes et al.)                   | 2025 | V-JEPA のスケールアップ版。3D RoPE を採用。本実装の設計基盤 |

## 表現学習手法（候補・ベースライン）

| 論文                                                                                                        | 年   | 備考                                               |
| ----------------------------------------------------------------------------------------------------------- | ---- | -------------------------------------------------- |
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling)                       | 2013 | VAE の原論文。ベースラインおよび候補手法として使用 |
| [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., CVPR 2022) | 2022 | MAE の原論文。候補手法                             |
