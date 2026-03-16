---
name: Hydra MISSING marker
description: Hydra設定で必須パラメータにはnullではなく???を使う
type: feedback
---

Hydraの設定ファイルで「必須パラメータ（Must be specified）」を表現する場合、`null` ではなく `???`（Hydra の MISSING マーカー）を使うこと。

**Why:** `null` だと実行時に `None` として通ってしまい、後段でエラーになる。`???` なら Hydra が設定解決時に未指定を検出してわかりやすいエラーを出す。

**How to apply:** Hydra config YAML で `# Must be specified` 等のコメントがある `null` 値を見つけたら `???` に置き換える。
