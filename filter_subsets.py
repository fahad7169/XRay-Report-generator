import argparse
import pandas as pd


DISEASE_KWS = [
    'pneumothorax', 'effusion', 'consolidation', 'atelectasis', 'edema',
    'cardiomegaly', 'opacity', 'infiltrate', 'nodule', 'mass', 'granuloma',
    'calcification', 'cabg', 'sternotomy', 'pleural', 'collapse', 'airspace'
]


def pick_text_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = c.lower()
        if lc in ('report', 'findings', 'impression', 'caption', 'text'):
            return c
    # fallback to last column
    return df.columns[-1]


def main():
    ap = argparse.ArgumentParser(description='Filter CSV into disease and normal subsets by keywords in report text.')
    ap.add_argument('--csv', required=True, help='Input CSV (e.g., Final_CV_Data.csv)')
    ap.add_argument('--out-normal', default='CV_normal_subset.csv', help='Output CSV for normal subset')
    ap.add_argument('--out-disease', default='CV_disease_subset.csv', help='Output CSV for disease subset')
    ap.add_argument('--normal-n', type=int, default=5, help='Number of normal rows to save')
    ap.add_argument('--disease-n', type=int, default=5, help='Number of disease rows to save')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding_errors='ignore')
    text_col = pick_text_column(df)
    tx = df[text_col].astype(str).str.lower().fillna('')

    disease_mask = tx.apply(lambda t: any(k in t for k in DISEASE_KWS))
    disease_df = df[disease_mask].head(args.disease_n)
    normal_df = df[~disease_mask].head(args.normal_n)

    disease_df.to_csv(args.out_disease, index=False)
    normal_df.to_csv(args.out_normal, index=False)

    print(f'Saved {len(normal_df)} normal rows -> {args.out_normal}')
    print(f'Saved {len(disease_df)} disease rows -> {args.out_disease}')


if __name__ == '__main__':
    main()


