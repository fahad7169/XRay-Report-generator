import argparse
import os
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


def pick_text_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ["report", "findings", "impression", "caption", "text"]:
            return c
    return df.columns[-1]


def main():
    ap = argparse.ArgumentParser(description="Export tokenizer.pkl from a CSV of reports.")
    ap.add_argument("--csv", default="Final_Train_Data.csv", help="Input CSV (training)")
    ap.add_argument("--out", default="models/tokenizer.pkl", help="Output tokenizer pickle path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.csv, encoding_errors="ignore")
    text_col = pick_text_column(df)
    texts = df[text_col].astype(str).tolist()

    tok = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
    tok.fit_on_texts(texts)

    with open(args.out, "wb") as f:
        pickle.dump(tok, f)
    print(f"Saved tokenizer -> {args.out}")


if __name__ == "__main__":
    main()


