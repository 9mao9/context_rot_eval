#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Rot 評価の実行スクリプト

使用方法:
    python run.py                    # デフォルト設定で実行
    python run.py --quick           # クイック評価（少ないテスト数）
    python run.py --full            # フル評価（多くのテスト数）
    python run.py --config <path>   # カスタム設定で実行
"""

import sys
import io

# Windows環境での文字化け対策
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import sys
from pathlib import Path
import numpy as np

# モジュールをインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from context_eval import ContextRotEvaluator, load_config


def main():
    parser = argparse.ArgumentParser(
        description="LLM Context Rot 評価スクリプト"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="設定ファイルのパス",
        default=None
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="standard",
        help="評価モード"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="モデルファイルのパス（設定を上書き）",
        default=None
    )
    
    args = parser.parse_args()
    
    # 設定をロード
    config_path = Path(__file__).parent / "config.yaml"
    if args.config:
        config_path = Path(args.config)
    
    config = load_config(str(config_path))
    
    # コマンドライン引数で設定を上書き
    if args.model:
        config["model_path"] = args.model
    
    # 評価エンジンを初期化
    evaluator = ContextRotEvaluator(config)
    
    # モデルをロード
    if not evaluator.load_model():
        print("エラー: モデルの読み込みに失敗しました")
        return 1
    
    # モード別のテスト設定
    if args.mode == "quick":
        print("クイック評価モード（テスト数は最小限）")
        ctx_lengths = [2000, 4000]
        positions = np.linspace(0, 1, 3)
    elif args.mode == "full":
        print("フル評価モード（詳細な測定）")
        ctx_lengths = [1000, 2000, 4000, 8000]
        positions = np.linspace(0, 1, 11)
    else:  # standard
        print("標準評価モード")
        ctx_lengths = [2000, 4000, 8000]
        positions = np.linspace(0, 1, 5)
    
    print(f"テスト設定: {len(ctx_lengths)} × {len(positions)} = {len(ctx_lengths) * len(positions)} テスト")
    
    # 評価を実行
    evaluator.run_evaluation(ctx_lengths, positions)
    
    # 結果を可視化
    print("\n可視化を生成中...")
    evaluator.generate_heatmap()
    evaluator.generate_decay_plot()
    
    # 結果を保存
    print("\n結果を保存中...")
    evaluator.save_results_json()
    evaluator.generate_summary_report()
    
    print("\n✓ 評価完了！")
    print(f"結果フォルダ: {evaluator.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
