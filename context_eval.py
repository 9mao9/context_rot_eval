"""
LLM Context Rot 評価スクリプト
仕様: context_rot.md に基づいて、LLMのコンテキスト劣化を測定します。
"""

import os
import sys
import io
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

# matplotlib で日本語フォント設定（Windows対応）
try:
    import japanize_matplotlib
except ImportError:
    pass

try:
    # Windows環境での日本語フォント設定
    import platform
    if platform.system() == 'Windows':
        matplotlib.rcParams['font.sans-serif'] = ['Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'DejaVu Sans']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'DejaVu Sans', 'Arial']
except:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

# プロジェクトのルートパスを設定
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("警告: llama-cpp-pythonがインストールされていません")


class ContextRotEvaluator:
    """Context Rot評価エンジン"""
    
    def __init__(self, config: Dict):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.model_path = config.get("model_path")
        self.max_ctx = config.get("max_context_length", 8192)
        self.needle = config.get("needle", "今日のラッキーアイテムは「青い定規」です。")
        self.retrieval_prompt = config.get("retrieval_prompt", 
                                          "コンテキストに基づき、今日のラッキーアイテムを答えてください。")
        self.llm = None
        self.results = []
        self.output_dir = Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model(self) -> bool:
        """モデルをロード"""
        if not LLAMA_AVAILABLE:
            print("エラー: llama-cpp-pythonが必要です")
            return False
            
        if not os.path.exists(self.model_path):
            print(f"エラー: モデルファイルが見つかりません: {self.model_path}")
            return False
            
        try:
            print(f"モデルをロード中: {self.model_path}")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.max_ctx,
                n_gpu_layers=-1,
                verbose=False
            )
            print("✓ モデルのロード完了")
            return True
        except Exception as e:
            print(f"エラー: モデルのロードに失敗: {e}")
            return False
    
    def build_haystack(self, target_tokens: int, position_pct: float) -> str:
        """
        Needleをhaystackに埋め込む
        
        Args:
            target_tokens: 目標トークン数
            position_pct: Needleの挿入位置（0.0-1.0）
            
        Returns:
            組み立てられたコンテキスト
        """
        # ダミーテキスト生成
        dummy_base = (
            "これは評価用のダミーテキストです。"
            "自然言語処理とLLMの応用について考察を深めることは重要です。"
            "データ分析と統計的手法は現代社会における情報理解の基本です。"
            "機械学習技術の発展により、多くの分野で自動化が進んでいます。"
            "コンテキストウィンドウの長さはLLMの性能に大きな影響を与えます。"
        )
        
        # 十分な長さのhaystackを作成
        repeat_count = max(1, target_tokens // len(dummy_base) + 1)
        current_haystack = dummy_base * repeat_count
        
        # トークン化（llama-cpp-pythonを使用）
        if self.llm:
            try:
                haystack_tokens = self.llm.tokenize(
                    current_haystack.encode('utf-8', errors='ignore')
                )
                needle_tokens = self.llm.tokenize(
                    self.needle.encode('utf-8', errors='ignore')
                )
                
                # 挿入位置を計算
                insert_idx = max(0, int(len(haystack_tokens) * position_pct))
                
                # 針を挿入
                combined_tokens = (
                    haystack_tokens[:insert_idx] + 
                    needle_tokens + 
                    haystack_tokens[insert_idx:]
                )
                
                # トークン数を目標に調整
                if len(combined_tokens) > target_tokens:
                    combined_tokens = combined_tokens[:target_tokens]
                else:
                    # 不足分を埋める
                    combined_tokens.extend(haystack_tokens[:target_tokens - len(combined_tokens)])
                
                context = self.llm.detokenize(combined_tokens).decode('utf-8', errors='ignore')
                return context
            except Exception as e:
                print(f"トークン化エラー: {e}")
                return current_haystack[:target_tokens]
        else:
            # llama_cppなしの場合は文字数で調整
            return current_haystack[:target_tokens]
    
    def evaluate_recall(self, context: str) -> float:
        """
        LLMに問い合わせて想起率を評価
        
        Args:
            context: 評価用コンテキスト
            
        Returns:
            想起率（0.0 or 1.0）
        """
        if not self.llm:
            return 0.0
            
        try:
            prompt = f"コンテキスト: {context}\n\n質問: {self.retrieval_prompt}\n答え:"
            
            output = self.llm(
                prompt,
                max_tokens=50,
                stop=["\n", "。"],
                temperature=0.1
            )
            
            response = output["choices"][0]["text"].strip()
            
            # Needle内の重要な単語の検出
            keywords = ["青い定規", "定規"]
            detected = any(keyword in response for keyword in keywords)
            
            return 1.0 if detected else 0.0
            
        except Exception as e:
            print(f"評価エラー: {e}")
            return 0.0
    
    def run_evaluation(self, ctx_lengths: List[int], positions: List[float]) -> None:
        """
        評価を実行
        
        Args:
            ctx_lengths: テストするコンテキスト長のリスト
            positions: テストするNeedle位置のリスト（0.0-1.0）
        """
        print("\n=" * 60)
        print("コンテキスト劣化評価を開始します")
        print("=" * 60)
        
        if not self.llm:
            print("エラー: モデルが読み込まれていません")
            return
        
        total_tests = len(ctx_lengths) * len(positions)
        current_test = 0
        
        for ctx_len in ctx_lengths:
            for pos in positions:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] "
                      f"コンテキスト長: {ctx_len:,} tokens, "
                      f"位置: {pos:.1%}")
                
                context = self.build_haystack(ctx_len, pos)
                recall = self.evaluate_recall(context)
                
                self.results.append({
                    "context_length": ctx_len,
                    "position": pos,
                    "recall": recall,
                    "context_chars": len(context)
                })
                
                print(f"   想起率: {'✓' if recall == 1.0 else '✗'} ({recall:.0%})")
    
    def calculate_half_life(self) -> float:
        """
        Rot Half-Lifeを計算
        
        Returns:
            想起率が50%に低下するコンテキスト長
        """
        if not self.results:
            return float('inf')
        
        # コンテキスト長ごとの平均想起率を計算
        ctx_recalls = {}
        for result in self.results:
            ctx_len = result["context_length"]
            if ctx_len not in ctx_recalls:
                ctx_recalls[ctx_len] = []
            ctx_recalls[ctx_len].append(result["recall"])
        
        x_data = np.array(sorted(ctx_recalls.keys()))
        y_data = np.array([np.mean(ctx_recalls[ctx]) for ctx in x_data])
        
        # 減衰関数でフィッティング
        def decay_func(L, r0, lam):
            return r0 * np.exp(-lam * L / 1000)  # 1000トークンで正規化
        
        try:
            popt, _ = curve_fit(decay_func, x_data, y_data, p0=[1.0, 0.001], maxfev=5000)
            r0_fit, lam_fit = popt
            
            if lam_fit > 0:
                half_life = np.log(2) / lam_fit * 1000
            else:
                half_life = float('inf')
                
            return half_life
        except Exception as e:
            print(f"フィッティングエラー: {e}")
            return float('inf')
    
    def generate_heatmap(self) -> None:
        """想起率のヒートマップを生成"""
        if not self.results:
            print("結果がありません")
            return
        
        # データを行列化
        ctx_lengths = sorted(set(r["context_length"] for r in self.results))
        positions = sorted(set(r["position"] for r in self.results))
        
        heatmap = np.zeros((len(positions), len(ctx_lengths)))
        
        for i, pos in enumerate(positions):
            for j, ctx_len in enumerate(ctx_lengths):
                matching = [r["recall"] for r in self.results 
                           if r["position"] == pos and r["context_length"] == ctx_len]
                heatmap[i][j] = matching[0] if matching else 0
        
        # プロット
        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(label='想起率', shrink=0.8)
        
        plt.xlabel('コンテキスト長 (tokens)', fontsize=11)
        plt.ylabel('Needle位置', fontsize=11)
        plt.title('コンテキスト劣化ヒートマップ (Needle in a Haystack)', fontsize=12)
        
        plt.xticks(range(len(ctx_lengths)), [f"{c:,}" for c in ctx_lengths], rotation=45, fontsize=10)
        plt.yticks(range(len(positions)), [f"{p:.0%}" for p in positions], fontsize=10)
        
        output_path = self.output_dir / "heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ ヒートマップを保存: {output_path}")
        plt.close()
    
    def generate_decay_plot(self) -> None:
        """減衰カーブを生成"""
        if not self.results:
            return
        
        # コンテキスト長ごとの平均想起率
        ctx_recalls = {}
        for result in self.results:
            ctx_len = result["context_length"]
            if ctx_len not in ctx_recalls:
                ctx_recalls[ctx_len] = []
            ctx_recalls[ctx_len].append(result["recall"])
        
        x_data = np.array(sorted(ctx_recalls.keys()))
        y_data = np.array([np.mean(ctx_recalls[ctx]) for ctx in x_data])
        
        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'bo-', markersize=8, linewidth=2, label='測定値')
        plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='50% しきい値')
        plt.axhline(y=0.9, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='90% 信頼限界')
        
        plt.xlabel('コンテキスト長 (tokens)', fontsize=12)
        plt.ylabel('想起率', fontsize=12)
        plt.title('コンテキスト劣化の減衰曲線', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.ylim(-0.05, 1.1)
        
        output_path = self.output_dir / "decay_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 減衰カーブを保存: {output_path}")
        plt.close()
    
    def save_results_json(self) -> None:
        """結果をJSON形式で保存"""
        half_life = self.calculate_half_life()
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": self.results,
            "summary": {
                "total_tests": len(self.results),
                "total_recall_rate": np.mean([r["recall"] for r in self.results]),
                "rot_half_life_tokens": half_life if not np.isinf(half_life) else None,
            }
        }
        
        output_path = self.output_dir / "results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 結果をJSON形式で保存: {output_path}")
    
    def generate_summary_report(self) -> None:
        """サマリーレポートを生成"""
        if not self.results:
            print("結果がありません")
            return
        
        half_life = self.calculate_half_life()
        
        # 統計情報を計算
        recalls = [r["recall"] for r in self.results]
        avg_recall = np.mean(recalls)
        
        # コンテキスト長ごとの統計
        ctx_recalls = {}
        for result in self.results:
            ctx_len = result["context_length"]
            if ctx_len not in ctx_recalls:
                ctx_recalls[ctx_len] = []
            ctx_recalls[ctx_len].append(result["recall"])
        
        report = []
        report.append("# コンテキスト劣化評価レポート")
        report.append("")
        report.append(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## 1. 評価設定")
        report.append("")
        report.append(f"- **モデルパス**: {self.model_path}")
        report.append(f"- **Needle**: {self.needle}")
        report.append(f"- **Needle検出キーワード**: 「青い定規」")
        report.append(f"- **最大コンテキスト長**: {self.max_ctx:,} tokens")
        report.append("")
        
        report.append("## 2. 評価結果概要")
        report.append("")
        report.append(f"- **総テスト数**: {len(self.results)}")
        report.append(f"- **平均想起率**: {avg_recall:.1%}")
        report.append(f"- **成功数**: {int(sum(recalls))}/{len(recalls)}")
        report.append("")
        
        if not np.isinf(half_life):
            report.append(f"### Rot Half-Life: **{half_life:,.0f} tokens**")
            report.append(f"想起率が50%に低下するコンテキスト長を意味します。")
        else:
            report.append("### Rot Half-Life: 計算不可")
        report.append("")
        
        report.append("## 3. コンテキスト長別の結果")
        report.append("")
        report.append("| コンテキスト長 | テスト数 | 想起率 |")
        report.append("|---------|---------|-------|")
        
        for ctx_len in sorted(ctx_recalls.keys()):
            recalls_for_ctx = ctx_recalls[ctx_len]
            avg = np.mean(recalls_for_ctx)
            report.append(
                f"| {ctx_len:,} | {len(recalls_for_ctx)} | {avg:.1%} |"
            )
        report.append("")
        
        report.append("## 4. 安全運用ガイドライン")
        report.append("")
        report.append("- **信頼可能限界 (90%)**: 想起率が90%以上の最大コンテキスト長")
        report.append("- **注意限界 (50%)**: Rot Half-Life (R(L) = 50%)")
        report.append("- **危険域 (<50%)**: コンテキスト長がこれを超える場合、情報損失の可能性あり")
        report.append("")
        
        report.append("## 5. 推奨事項")
        report.append("")
        report.append("- 短期タスク（<4K tokens）: 安定した想起率が期待できます")
        report.append("- 中期タスク (4K～8K tokens): 位置による想起率のバラツキに注意")
        report.append("- 長期タスク (>8K tokens): RAG、要約、またはコンテキストローテーション導入を検討")
        report.append("")
        
        report_text = "\n".join(report)
        
        output_path = self.output_dir / "SUMMARY.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ サマリーレポートを保存: {output_path}")
        
        # コンソール出力
        print("\n" + "=" * 60)
        print("評価結果サマリー")
        print("=" * 60)
        print(report_text)


def load_config(config_path: str) -> Dict:
    """設定ファイルをロード"""
    if not os.path.exists(config_path):
        # デフォルト設定を返す
        return {
            "model_path": str(PROJECT_ROOT / "user_data" / "models" / "Umievo-itr012-Gleipnir-7B" / "Umievo-itr012-Gleipnir-7B.Q4_K_M.gguf"),
            "max_context_length": 4096,
            "needle": "今日のラッキーアイテムは「青い定規」です。",
            "retrieval_prompt": "コンテキストに基づき、今日のラッキーアイテムを答えてください。一言で答えてください。",
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        import yaml
        return yaml.safe_load(f)


if __name__ == "__main__":
    # 設定をロード
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # 評価エンジンを初期化
    evaluator = ContextRotEvaluator(config)
    
    # モデルをロード
    if not evaluator.load_model():
        print("モデルの読み込みに失敗しました")
        exit(1)
    
    # 評価を実行
    ctx_lengths = [2000, 4000]  # 簡易版
    positions = np.linspace(0, 1, 3)  # 3ポイント
    
    evaluator.run_evaluation(ctx_lengths, positions)
    
    # 結果を可視化
    evaluator.generate_heatmap()
    evaluator.generate_decay_plot()
    
    # 結果を保存
    evaluator.save_results_json()
    evaluator.generate_summary_report()
    
    print("\n✓ 評価完了！")
