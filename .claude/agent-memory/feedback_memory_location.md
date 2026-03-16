---
name: Memory location preference
description: メモリはプロジェクト内の.claude/agent-memory/に保存する
type: feedback
---

メモリファイルはプロジェクト内の `/workspace/.claude/agent-memory/` に保存すること。`/root/.claude/projects/` 側ではなくリポジトリ内に置く。

**Why:** プロジェクトと一緒にバージョン管理・共有できるようにするため。

**How to apply:** 新しいメモリを保存する際は `/workspace/.claude/agent-memory/` に書き、MEMORY.md のインデックスもそちらを参照する。
