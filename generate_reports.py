import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2


def load_chexnet_feature_extractor(weights_path: str) -> Model:
    base = densenet.DenseNet121(
        include_top=False, weights=None, input_shape=(224, 224, 3), pooling="avg"
    )
    out = Dense(14, activation="sigmoid", name="predictions")(base.output)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(weights_path)
    # second to last layer is global pooled features
    feat_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feat_model


def load_image_as_model_input(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def infer_image_features(chexnet_model: Model, img1_path: str, img2_path: str) -> np.ndarray:
    i1 = load_image_as_model_input(img1_path)
    i2 = load_image_as_model_input(img2_path)
    f1 = chexnet_model.predict(i1, verbose=0)
    f2 = chexnet_model.predict(i2, verbose=0)
    return np.concatenate((f1, f2), axis=1)


def _top_k_logits(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= probs.size:
        return probs
    idx = np.argpartition(probs, -k)[-k:]
    masked = np.zeros_like(probs)
    masked[idx] = probs[idx]
    return masked


def _sample_from_probs(probs: np.ndarray, temperature: float = 1.0, top_k: int = 0) -> int:
    p = probs.astype("float64")
    p = np.maximum(p, 1e-9)
    if temperature and temperature != 1.0:
        p = np.power(p, 1.0 / temperature)
    if top_k and top_k > 0:
        p = _top_k_logits(p, top_k)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


def generate_report(
    encoder_model: tf.keras.Model,
    decoder_model: tf.keras.Model,
    tokenizer,
    image_features: np.ndarray,
    max_len: int = 153,
    top_k: int = 5,
    temperature: float = 0.8,
) -> str:
    end_id = tokenizer.word_index.get("endseq")
    start_id = tokenizer.word_index.get("startseq")
    if end_id is None or start_id is None:
        raise RuntimeError("startseq/endseq not in tokenizer vocabulary.")

    input_tokens = [start_id]
    words: List[str] = []
    for _ in range(max_len):
        inp = pad_sequences([input_tokens], max_len, padding="post")
        preds = decoder_model.predict([inp, image_features], verbose=0)
        next_id = _sample_from_probs(preds[0], temperature=temperature, top_k=top_k)
        if next_id == end_id or next_id == 0 or next_id not in tokenizer.index_word:
            break
        words.append(tokenizer.index_word[next_id])
        input_tokens.append(next_id)
    return " ".join(words)


def infer_for_rows(
    df: pd.DataFrame,
    id_col: str,
    img1_col: str,
    img2_col: str,
    encoder_model: tf.keras.Model,
    decoder_model: tf.keras.Model,
    tokenizer,
    chexnet_model: tf.keras.Model,
    n: int,
) -> List[Tuple[str, str]]:
    results = []
    for _, row in df.head(n).iterrows():
        pid = str(row[id_col])
        img1 = str(row[img1_col])
        img2 = str(row[img2_col])
        feats = infer_image_features(chexnet_model, img1, img2)
        rep = generate_report(encoder_model, decoder_model, tokenizer, feats)
        results.append((pid, rep))
    return results


def guess_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    # id column
    id_candidates = [c for c in df.columns if c.lower() in ("person_id", "key", "id")]
    id_col = id_candidates[0] if id_candidates else df.columns[0]
    # image columns: pick first two columns whose value looks like a png path
    img_cols = []
    for c in df.columns:
        v = str(df[c].iloc[0]) if len(df) else ""
        if v.endswith(".png"):
            img_cols.append(c)
    if len(img_cols) < 2:
        # fallback: look for columns containing 'png' anywhere
        img_cols = [c for c in df.columns if "png" in str(df[c].astype(str).head(20).tolist()).lower()][:2]
    if len(img_cols) < 2:
        raise ValueError("Could not infer two image columns. Use --img1-col/--img2-col.")
    return id_col, img_cols[0], img_cols[1]


def main():
    ap = argparse.ArgumentParser(description="Generate radiology reports using saved encoder/decoder models.")
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., Final_CV_Data.csv)")
    ap.add_argument("--models-dir", default="models", help="Directory with saved models and tokenizer.pkl")
    ap.add_argument("--chexnet-weights", default="brucechou1983_CheXNet_Keras_0.3.0_weights.h5", help="CheXNet weights .h5 path")
    ap.add_argument("--id-col", default=None, help="ID column name (default: auto-detect)")
    ap.add_argument("--img1-col", default=None, help="First image column (default: auto-detect)")
    ap.add_argument("--img2-col", default=None, help="Second image column (default: auto-detect)")
    ap.add_argument("--n", type=int, default=5, help="Number of rows to infer")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k sampling")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = ap.parse_args()

    # Load artifacts
    enc_path = os.path.join(args.models_dir, "encoder_model")
    dec_path = os.path.join(args.models_dir, "decoder_model")
    tok_path = os.path.join(args.models_dir, "tokenizer.pkl")
    if not (os.path.exists(enc_path) and os.path.exists(dec_path) and os.path.exists(tok_path)):
        raise FileNotFoundError(
            "Missing models/tokenizer. From the notebook, save: encoder_model.save('models/encoder_model'); "
            "decoder_model.save('models/decoder_model'); pickle.dump(tokenizer, open('models/tokenizer.pkl','wb'))"
        )

    print("Loading models and tokenizer...")
    encoder_model = tf.keras.models.load_model(enc_path)
    decoder_model = tf.keras.models.load_model(dec_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    print("Building CheXNet feature extractor...")
    chexnet_model = load_chexnet_feature_extractor(args.chexnet_weights)

    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    id_col = args.id_col
    img1_col = args.img1_col
    img2_col = args.img2_col
    if not (id_col and img1_col and img2_col):
        id_col, img1_col, img2_col = guess_columns(df)
    print(f"Using columns -> id: {id_col}, img1: {img1_col}, img2: {img2_col}")

    # Monkey-patch generation settings via closure defaults
    def gen(feats: np.ndarray) -> str:
        return generate_report(
            encoder_model, decoder_model, tokenizer, feats, top_k=args.top_k, temperature=args.temperature
        )

    results = []
    for _, row in df.head(args.n).iterrows():
        pid = str(row[id_col])
        img1 = str(row[img1_col])
        img2 = str(row[img2_col])
        feats = infer_image_features(chexnet_model, img1, img2)
        rep = gen(feats)
        results.append((pid, rep))
        print("------------------------------------------------------------------------------------------------------")
        print(f"ID: {pid}")
        print(f"Predicted Report: {rep}")

    print("\nDone.")


if __name__ == "__main__":
    main()


