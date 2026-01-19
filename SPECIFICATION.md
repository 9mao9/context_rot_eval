# Context Rot 評価仕様の概要

## 仕様の翻訳

### 1. Needle Recall vs Position (NIAH)

**仕様から**: 膨大な無関係なテキスト（Haystack）の中に、特定の事実（Needle）を埋め込み、その抽出精度を測定します。

**実装**:
- `ContextRotEvaluator.build_haystack()`: Needleを指定位置にhaystackに埋め込む
- `ContextRotEvaluator.evaluate_recall()`: LLMに問い合わせて、Needleが検出されたかを判定（0.0 or 1.0）
- `ContextRotEvaluator.generate_heatmap()`: 結果をヒートマップで可視化

**パラメータ**:
- x軸（Needle位置）: 0.0（文頭）～ 1.0（文末）
- y軸（コンテキスト長）: トークン数

### 2. Rot Half-Life (H_L)

**仕様から**: 想起率が初期値の50%まで低下するコンテキスト長を算出

```
減衰モデル: R(L) = R₀ × e^(-λL)
Half-Life: H_L = ln(2) / λ ≈ 0.693 / λ
```

**実装**:
- `ContextRotEvaluator.calculate_half_life()`: scipy.optimize.curve_fitを使用してパラメータを推定
- 減衰関数 `decay_func(L, r0, lam) = r0 * exp(-lam * L / 1000)`
- 1000トークン単位で正規化して、より安定したフィッティングを実現

**計算ロジック**:
1. コンテキスト長ごとに平均想起率を計算
2. 指数関数でフィッティング
3. λから Half-Life を算出

## テストの流れ

### ステップ 1: 準備フェーズ

```python
# Needleの設定（学習データに含まれない固有事実）
needle = "今日のラッキーアイテムは「青い定規」です。"

# Haystack（ダミーテキスト）の生成
# - モデルの学習傾向に類似した自然な日本語テキスト
# - 十分に長いテキスト（目標トークン数以上）
```

### ステップ 2: テスト実行フェーズ

各 (コンテキスト長, Needle位置) の組み合わせについて：

```python
# 1. コンテキストを構築
context = build_haystack(
    target_tokens=ctx_length,
    position_pct=position
)

# 2. LLMに問い合わせ
response = llm.generate(
    prompt=f"Context: {context}\nQuestion: {retrieval_prompt}",
    max_tokens=50
)

# 3. 想起率を評価
recall = 1.0 if "青い定規" in response else 0.0

# 4. 結果を記録
results.append({
    "context_length": ctx_length,
    "position": position,
    "recall": recall
})
```

### ステップ 3: 分析フェーズ

```python
# 1. コンテキスト長ごとの平均想起率を計算
ctx_recalls = {ctx_len: mean([r["recall"] for r in results if r["ctx_len"] == ctx_len])}

# 2. 減衰曲線をフィッティング
half_life = calculate_half_life(ctx_recalls)

# 3. 可視化
- heatmap: (position, ctx_length) → recall
- decay_curve: ctx_length → avg_recall
```

## 評価の解釈

### 想起率の意味

| 想起率 | 解釈 | 原因 |
|-------|------|------|
| 1.0 (100%) | Needleを正しく検出 | ✓ コンテキストが短い、または位置が有利 |
| 0.0 (0%) | Needleを検出できず | ✗ コンテキストが長い、または位置が不利 |

### Half-Lifeの解釈

**例**: Half-Life = 6500 tokens

これは、以下を意味します：

```
- 2000 tokens: R ≈ 90%
- 4000 tokens: R ≈ 80%
- 6500 tokens: R ≈ 50%  ← Half-Life
- 13000 tokens: R ≈ 25%
```

### 位置による想起率の変化

ヒートマップで3つのパターンが観察されます：

#### パターン1: 中央崩壊（Middle Collapse）
```
位置: 0%  25%  50%  75%  100%
想起率: ✓  ✗  ✗  ✓  ✓
```
→ 中央に埋め込まれた情報が忘れられやすい

#### パターン2: 末尾優先（Recency Bias）
```
位置: 0%  25%  50%  75%  100%
想起率: ✗  ✗  ✗  ✓  ✓
```
→ 最新の情報（末尾）を優先的に記憶

#### パターン3: 安定（Stable）
```
位置: 0%  25%  50%  75%  100%
想起率: ✓  ✓  ✓  ✓  ✓
```
→ 位置に関わらず安定した想起率

## 実装上の設計選択

### 1. Needle と Haystack の選択

**Needle**: 
- ユニークな事実（学習データに含まれない）
- 固定長の短いテキスト
- 検出キーワードが明確

**Haystack**:
- 自然言語テキスト（Wikipediaのダミーデータなど）
- 意味のある内容（ノイズではない）
- 十分に長い

### 2. トークン化の手法

```python
# llama-cpp-pythonを使用してトークン化
tokens = llm.tokenize(text.encode('utf-8'))

# 利点:
# - 実際のモデルと同じトークナイザーを使用
# - 正確なトークン数の計測
```

### 3. 想起率の評価方法

```python
# キーワード検出による二値判定
recall = 1.0 if "青い定規" in response else 0.0

# 理由:
# - シンプルで再現可能
# - ノイズに強い（部分的な回答も許容）
```

### 4. 減衰関数のモデル

```python
# 指数減衰: R(L) = R₀ × e^(-λL)
# 理由:
# - 物理的・生物学的に自然
# - パラメータが少ない（解釈しやすい）
```

## 出力ファイルの詳細

### results.json

```json
{
  "timestamp": "ISO 8601形式の実行時刻",
  "config": {
    "model_path": "使用したモデル",
    "max_context_length": "設定最大長",
    "needle": "埋め込まれたテキスト"
  },
  "results": [
    {
      "context_length": コンテキスト長(tokens),
      "position": Needle位置(0.0-1.0),
      "recall": 想起率(0.0 or 1.0),
      "context_chars": コンテキストの文字数
    }
  ],
  "summary": {
    "total_tests": テスト総数,
    "total_recall_rate": 全体の平均想起率,
    "rot_half_life_tokens": Rot Half-Life(tokens)
  }
}
```

### SUMMARY.md

マークダウン形式のレポート：
1. 評価設定の確認
2. 結果概要（総テスト数、平均想起率、成功数）
3. **Rot Half-Life の値（重要！）**
4. コンテキスト長別の結果表
5. 安全運用ガイドライン
6. 推奨事項

### グラフ

#### heatmap.png
```
X軸: コンテキスト長 (2000, 4000, 8000 tokens)
Y軸: Needle位置 (0%, 25%, 50%, 75%, 100%)
色: 想起率
  - 緑 (1.0): 想起成功
  - 赤 (0.0): 想起失敗
```

#### decay_curve.png
```
X軸: コンテキスト長 (tokens)
Y軸: 想起率 (0.0-1.0)

プロット内容:
- 青い線: 測定値（各コンテキスト長での平均想起率）
- 赤い破線: 50% しきい値（Half-Life）
- オレンジ破線: 90% 信頼限界
```

## 運用ガイドライン

### 信頼可能な最大入力長の決定

| 想起率 | 運用レベル | 推奨事項 |
|-------|---------|--------|
| > 90% | 安全 | そのまま使用可能 |
| 50%～90% | 注意 | 重要情報は複製、圧縮を検討 |
| < 50% | 危険 | RAG、要約、またはチャンク分割を導入 |

### RAG（Retrieval-Augmented Generation）の導入基準

Half-Life が 4K tokens 以下の場合：
```
1. コンテキストを固定サイズのチャンクに分割
2. 重要なチャンクのみを選択
3. LLMに提供
```

### コンテキストローテーションの導入基準

Half-Life が 2K tokens 以下の場合：
```
1. 長文を短い段落に分割
2. 各段落ごとに分析を実行
3. 結果を統合
```

## トラブルシューティング

### 想起率が常に 0% の場合

1. Needleが正しく埋め込まれているか確認
2. LLMの生成に問題がないか確認
3. キーワード検出ロジックを確認

```python
# デバッグ出力を追加
context = build_haystack(2000, 0.5)
print(f"コンテキスト長: {len(context)} 文字")
print(f"Needleが含まれている: {'青い定規' in context}")
```

### Half-Lifeが計算できない場合

```
"Rot Half-Life: 計算不可"
```

原因:
- テスト数が少なすぎる
- 想起率がすべて 0% または 100%
- モデルが不安定

対策:
- テスト数を増やす
- フル評価モードを使用
- 異なるNeedleを試す

---

**最終更新**: 2026年1月19日
