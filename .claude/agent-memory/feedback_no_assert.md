---
name: No assert statements
description: assertは使わない。型の絞り込みにはtype ignoreやcastを使う
type: feedback
---

assertは重いので使わないこと。

**Why:** assertはパフォーマンスコストがあり、`-O`フラグで無効化される。

**How to apply:**

- そもそも型アノテーションを正しくすることでassertやcastが不要な設計にする（例: `nn.Module` ではなく具体的な型を引数に指定）
- 文脈的に型が保証される場合のみ `typing.cast` を使う（`# type: ignore` より `cast` を優先）
- ランタイムチェックが本当に必要なら明示的な `if` + `raise` を使う
