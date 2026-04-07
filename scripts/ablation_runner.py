"""Small driver to run ablation experiments by invoking train_specrnet.py for multiple variants and seeds.
Usage example:
    python scripts/ablation_runner.py --variants full no-att gap --seeds 42 2023 7
"""
import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_variant(variant, seed, args):
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
    if args.weighted_loss:
        train_cmd.append('--weighted_loss')

    print('\n=== Running training:', ' '.join(train_cmd))
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {variant} seed {seed}: {e}")
        return

    # After training, run evaluation if checkpoint exists
    ckpt = Path(args.out_dir) / variant / f"seed_{seed}" / f"best_specrnet_{variant}_seed{seed}.pt"
    if not ckpt.exists():
        print(f"Checkpoint not found at {ckpt}, skipping evaluation.")
        return

    eval_out = Path(args.out_dir) / variant / f"seed_{seed}" / 'eval'
    eval_out.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        'python',
        str(ROOT / 'evaluate.py'),
        '--checkpoint', str(ckpt),
        '--metadata', args.metadata,
        '--features', args.features,
        '--split', args.split,
        '--output_dir', str(eval_out),
        '--variant', variant,
        '--seed', str(seed),
    ]
    print('--- Running evaluation:', ' '.join(eval_cmd))
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {variant} seed {seed}: {e}")
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['default', 'no-att', 'gru1', 'gap'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2023, 7])
    parser.add_argument('--metadata', type=str, default='metadata_multi.csv')
    parser.add_argument('--features', type=str, default='lfcc_features')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate results after runs')
    args = parser.parse_args()

    start = time.time()
    for v in args.variants:
        for s in args.seeds:
            run_variant(v, s, args)

    if args.aggregate:
        print('\n=== Aggregating results...')
        agg_cmd = ['python', str(ROOT / 'scripts' / 'aggregate_results.py'), '--results_dir', args.out_dir, '--out_csv', str(Path(args.out_dir) / 'ablation_summary.csv')]
        subprocess.run(agg_cmd, check=True)

    print(f"\nAll runs finished in {time.time() - start:.1f}s")


if __name__ == '__main__':
    main()
