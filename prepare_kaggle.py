"""Small driver to run ablation experiments by invoking train_specrnet.py for multiple variants and seeds.
"""
import argparse
import subprocess
import time
from pathlib import Path

# 根据你存放此脚本的位置确定根目录 (如果放在 scripts/ 下，parents[1]就是项目根目录)
ROOT = Path(__file__).resolve().parents[1] 

def run_variant(variant, seed, args):
    # 1. 训练命令
    train_cmd = [
        'python',
        str(ROOT / 'train_specrnet.py'),
        '--variant', variant,
        '--seed', str(seed),
        '--metadata', args.metadata,
        '--features', args.features,
        '--out_dir', args.out_dir,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--scheduler', args.scheduler,
    ]
    
    print('\n' + '='*50)
    print(f'=== [TRAIN] Running variant: {variant} | seed: {seed} ===')
    print(' '.join(train_cmd))
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {variant} seed {seed}: {e}")
        return

    # 2. 自动评估命令 (新增！训练完直接测)
    best_model_path = Path(args.out_dir) / variant / f"seed_{seed}" / f"best_specrnet_{variant}_seed{seed}.pt"
    eval_out_dir = Path(args.out_dir) / f"eval_test_{variant}_seed{seed}"
    
    eval_cmd = [
        'python',
        str(ROOT / 'evaluate.py'),
        '--checkpoint', str(best_model_path),
        '--metadata', args.metadata,
        '--features', args.features,
        '--split', args.split,
        '--output_dir', str(eval_out_dir)
    ]
    
    print(f'\n=== [EVAL] Testing variant: {variant} | seed: {seed} ===')
    print(' '.join(eval_cmd))
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed for {variant} seed {seed}: {e}")

def main():
    parser = argparse.ArgumentParser()
    # 默认跑最后剩下的两个消融变体
    parser.add_argument('--variants', nargs='+', default=['no-att', 'gap'])
    # 默认跑一个随机种子 42 节省时间
    parser.add_argument('--seeds', nargs='+', type=int, default=[42]) 
    
    # 统一换成 Kaggle 平衡版的数据路径
    parser.add_argument('--metadata', type=str, default='metadata_kaggle.csv')
    parser.add_argument('--features', type=str, default='lfcc_features_kaggle')
    parser.add_argument('--out_dir', type=str, default='results_kaggle_balanced')
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--split', type=str, default='test')
    
    args = parser.parse_args()

    start = time.time()
    for v in args.variants:
        for s in args.seeds:
            run_variant(v, s, args)

    print(f"\n✅ All ablation experiments finished in {(time.time() - start)/60:.2f} minutes!")

if __name__ == '__main__':
    main()