# Context Rot 評価スイート

LLMがコンテキスト長の増大に伴い、情報をどの程度正確に保持・抽出できるかを測定するための評価ツールセットです。

## 概要

このツールは以下の指標を測定します：

### 1. Needle Recall vs Position (NIAH)
「干し草の中の針」テスト。膨大な無関係なテキスト（Haystack）の中に、特定の事実（Needle）を埋め込み、その抽出精度を測定します。

- **X軸**: Needleの挿入位置（0% = 文頭, 100% = 文末）
- **Y軸**: 総コンテキスト長（トークン数）
- **出力**: ヒートマップ形式の想起率（Recall Rate）

### 2. Rot Half-Life (H_L)
コンテキスト長が増加するにつれて想起率が低下する現象を「減衰」と捉え、想起率が初期値の50%まで低下するコンテキスト長を算出します。

```
R(L) = R₀ × e^(-λL)

H_L = ln(2) / λ
```

## ファイル構成

```
context_eval/
├── context_eval.py      # メイン評価エンジン
├── run.py              # 実行スクリプト
├── config.yaml         # 設定ファイル
├── requirements.txt    # 依存パッケージ
├── README.md           # このファイル
└── results/            # 評価結果の出力フォルダ
    ├── results.json    # 詳細な評価結果（JSON）
    ├── SUMMARY.md      # サマリーレポート（Markdown）
    ├── heatmap.png     # 想起率ヒートマップ
    └── decay_curve.png # 減衰カーブのグラフ
```

## インストール

### 前提条件
- Python 3.8+
- llama-cpp-python
- numpy, scipy, matplotlib

### 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な実行

```bash
python run.py
```

### 評価モード

3つの評価モードから選択できます：

#### 1. クイックモード（最速）
```bash
python run.py --mode quick
```
- テスト数: 2 × 3 = 6テスト
- 用途: 動作確認、デバッグ

#### 2. 標準モード（推奨）
```bash
python run.py --mode standard
```
- テスト数: 3 × 5 = 15テスト
- 用途: 一般的な評価

#### 3. フルモード（最詳細）
```bash
python run.py --mode full
```
- テスト数: 4 × 11 = 44テスト
- 用途: 詳細な分析、研究

### カスタム設定での実行

```bash
python run.py --config my_config.yaml
```

### モデルパスの指定

```bash
python run.py --model /path/to/model.gguf
```

## 設定ファイル (config.yaml)

### 主な設定項目

```yaml
# モデルパス
model_path: "{gguf model path}"

# 最大コンテキスト長
max_context_length: 4096

# Needle（埋め込むテキスト）
needle: "今日のラッキーアイテムは「青い定規」です。"

# 検索プロンプト
retrieval_prompt: "コンテキストに基づき、今日のラッキーアイテムを答えてください。"

# テスト設定
test_config:
  context_lengths: [2000, 4000, 8000]
  needle_positions: [0.0, 0.25, 0.5, 0.75, 1.0]
  temperature: 0.1
  max_tokens: 50
```

## 出力結果

### 1. results.json
詳細な評価結果をJSON形式で保存します：

```json
{
  "timestamp": "2026-01-19T12:00:00",
  "config": { /* 設定情報 */ },
  "results": [
    {
      "context_length": 2000,
      "position": 0.0,
      "recall": 1.0,
      "context_chars": 8234
    },
    ...
  ],
  "summary": {
    "total_tests": 15,
    "total_recall_rate": 0.87,
    "rot_half_life_tokens": 6500
  }
}
```

### 2. SUMMARY.md
評価結果のサマリーレポート（Markdown形式）：

- 評価設定の概要
- 総体的な結果統計
- Rot Half-Lifeの値
- コンテキスト長別の詳細結果
- 安全運用ガイドライン

### 3. heatmap.png
想起率をヒートマップで可視化：

- X軸: コンテキスト長（トークン数）
- Y軸: Needle位置（0% ～ 100%）
- 色: 想起率（緑=成功, 赤=失敗）

### 4. decay_curve.png
コンテキスト長に対する想起率の減衰曲線：

- 青い線: 測定値
- 赤い破線: 50% しきい値
- オレンジの破線: 90% 信頼限界

## 解釈ガイド

### Rot Half-Life の値

| Half-Life | 意味 | 推奨使用方法 |
|-----------|------|-----------|
| < 2K tokens | 劣化が速い | RAGやコンテキスト圧縮の導入が必須 |
| 2K～4K tokens | 中程度の劣化 | 4K以上のコンテキストでは注意 |
| 4K～8K tokens | 劣化が遅い | 一般的な用途では問題なし |
| > 8K tokens | 非常に安定 | 長文タスクでも安定 |

### 位置による想起率の変化

ヒートマップで注目すべき点：

- **中央部での低下**: Needleが中央に埋め込まれたときに想起率が低下する場合、「中央崩壊」と呼ぶ
- **末尾での低下**: コンテキストの終端での想起率低下は、最新情報への注意不足を示唆
- **バラツキ**: 位置による想起率のバラツキが大きいほど、安定性が低い

## トラブルシューティング

### モデルが見つからない
```
エラー: モデルファイルが見つかりません
```
→ config.yaml の `model_path` を確認してください。

### メモリ不足エラー
```
RuntimeError: Input buffer exceeded maximum size
```
→ config.yaml の `max_context_length` を減らしてください。

### llama-cpp-python が見つからない
```
ImportError: No module named 'llama_cpp'
```
→ `pip install -r requirements.txt` を実行してください。

## パフォーマンス最適化

### 高速化のためのヒント

1. **クイックモードを使用**
   ```bash
   python run.py --mode quick
   ```

2. **GPU アクセラレーションの有効化**
   - `context_eval.py` の `n_gpu_layers=-1` がGPU使用を有効にしています

3. **コンテキスト長を削減**
   - config.yaml で小さいコンテキスト長リストを設定

## 拡張・カスタマイズ

### カスタムNeedleの使用

config.yaml を編集して、異なるNeedleを指定できます：

```yaml
needle: "カスタムテキスト"
retrieval_prompt: "カスタムプロンプト"
```

### 異なるモデルでのテスト

複数モデルの比較評価：

```bash
python run.py --model model1.gguf --mode quick
python run.py --model model2.gguf --mode quick
```

## 参考文献

- LLaMA: Open and Efficient Foundation Language Models
- "Lost in the Middle: How Language Models Use Long Contexts" - Liu et al.
- Needle in a Haystack - LLM Context Length Evaluator

## ライセンス

License: MIT This evaluation was inspired by the "Needle In A Haystack" test (gkamradt) and the "Lost in the Middle" paper.

## サポート

問題や質問がある場合は、以下を確認してください：

1. config.yaml が正しく設定されているか
2. モデルファイルが存在するか
3. 必要なパッケージがすべてインストールされているか

---

**最終更新**: 2026年1月19日
