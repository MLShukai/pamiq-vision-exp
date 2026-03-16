---
name: No assert statements
description: assertは使わない。型の絞り込みにはtype: ignoreやcastを使う
type: feedback
---

assertは重いので使わないこと。

**Why:** assertはパフォーマンスコストがあり、`-O`フラグで無効化される。

**How to apply:** 型チェッカーの型絞り込みが必要な場合は `# type: ignore` や `typing.cast` を使う。ランタイムチェックが本当に必要なら明示的な `if` + `raise` を使う。
