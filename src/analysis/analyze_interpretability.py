"""
Compare consolidated interpretability outputs between two runs
Defaults: v2.11_none vs v2.11_minmax consolidated directories.
Reports overlap and coverage shifts in consensus biomarkers.
"""

from pathlib import Path
import pandas as pd


def load_consensus(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "consensus_biomarkers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing consensus_biomarkers.csv in {run_dir}")
    df = pd.read_csv(path)
    return df


def summarize(df: pd.DataFrame, label: str, top_k: int = 20):
    print(f"\n--- {label} ---")
    print(f"Models contributing: {df['n_models'].max():.0f}? (max in table)")
    print(f"Rows: {len(df)}; Top-{top_k} preview:")
    preview = df.head(top_k)[['feature', 'consensus_score', 'pct_models']]
    for _, row in preview.iterrows():
        print(f"  {row['feature']:12s} score={row['consensus_score']:.3f} coverage={row['pct_models']:.1f}%")


def overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int = 20):
    top_a = set(df_a.head(k)['feature'])
    top_b = set(df_b.head(k)['feature'])
    common = top_a & top_b
    return common, top_a, top_b


def coverage_stats(df: pd.DataFrame, k: int = 20):
    top = df.head(k)
    return {
        'mean_cov': top['pct_models'].mean(),
        'max_cov': top['pct_models'].max(),
        'min_cov': top['pct_models'].min(),
    }


def main():
    run_none = Path('results/interpretability/v2.11_none/consolidated')
    run_minmax = Path('results/interpretability/v2.11_minmax/consolidated')

    df_none = load_consensus(run_none)
    df_minmax = load_consensus(run_minmax)

    print("=" * 80)
    print("CONSOLIDATED INTERPRETABILITY COMPARISON")
    print("v2.11_none vs v2.11_minmax")
    print("=" * 80)

    summarize(df_none, "v2.11_none (none)")
    summarize(df_minmax, "v2.11_minmax (minmax)")

    common, top_none, top_min = overlap(df_none, df_minmax, k=20)
    print("\nTop-20 overlap: {}/20".format(len(common)))
    if common:
        print("  Shared:", ", ".join(sorted(common)))

    cov_none = coverage_stats(df_none, k=20)
    cov_min = coverage_stats(df_minmax, k=20)
    print("\nCoverage (mean/max/min) among top-20:")
    print(f"  none   : {cov_none['mean_cov']:.1f}% / {cov_none['max_cov']:.1f}% / {cov_none['min_cov']:.1f}%")
    print(f"  minmax : {cov_min['mean_cov']:.1f}% / {cov_min['max_cov']:.1f}% / {cov_min['min_cov']:.1f}%")

    # Features that appear in top-20 of one run but not the other
    only_none = top_none - top_min
    only_min = top_min - top_none
    if only_none:
        print("\nUnique to none (top-20):", ", ".join(sorted(only_none)))
    if only_min:
        print("Unique to minmax (top-20):", ", ".join(sorted(only_min)))

    # Quick check of consensus score shift for shared features
    if common:
        merged = (
            df_none[['feature', 'consensus_score', 'pct_models']].rename(columns={
                'consensus_score': 'score_none', 'pct_models': 'cov_none'
            })
            .merge(
                df_minmax[['feature', 'consensus_score', 'pct_models']].rename(columns={
                    'consensus_score': 'score_min', 'pct_models': 'cov_min'
                }),
                on='feature', how='inner'
            )
            .query('feature in @common')
        )
        print("\nShared feature score deltas (top-20 sets):")
        for _, row in merged.iterrows():
            delta = row['score_min'] - row['score_none']
            print(
                f"  {row['feature']:12s} Δscore={delta:+.3f} "
                f"cov none={row['cov_none']:.1f}% → min={row['cov_min']:.1f}%"
            )


if __name__ == "__main__":
    main()
