import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2


def build_encoder_decoder(vocab_size: int, max_len: int = 153) -> tf.keras.Model:
    # Image branch
    image_input = Input(shape=(2048,), name='Image_input')
    dense_encoder = Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=56), name='dense_encoder')(image_input)

    # Text branch
    text_input = Input(shape=(max_len,), name='Text_Input')
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=max_len,
                                mask_zero=True, trainable=False, name='Embedding_layer')(text_input)
    # Two-layer LSTM as commonly used; names must match saved weights
    lstm1 = LSTM(256, return_sequences=True, name='LSTM1')(embedding_layer)
    lstm2 = LSTM(256, name='LSTM2')(lstm1)
    dropout1 = Dropout(0.4, name='dropout1')(lstm2)

    add = Add(name='add')([dense_encoder, dropout1])
    fc1 = Dense(256, activation='relu', name='fc1')(add)
    dropout2 = Dropout(0.4, name='dropout2')(fc1)
    output_layer = Dense(vocab_size, activation='softmax', name='Output_layer')(dropout2)

    model = Model(inputs=[image_input, text_input], outputs=output_layer, name='encoder_decoder')
    return model


def build_encoder_decoder_inference_parts(encoder_decoder: tf.keras.Model) -> (tf.keras.Model, tf.keras.Model):
    # Encoder: from Image_input to dense_encoder output
    encoder_input = encoder_decoder.input[0]
    encoder_output = encoder_decoder.get_layer('dense_encoder').output
    encoder_model = Model(encoder_input, encoder_output)

    # Decoder: takes text_input and an external enc_output (256)
    text_input = encoder_decoder.input[1]
    enc_output = Input(shape=(256,), name='Enc_Output')
    text_output = encoder_decoder.get_layer('LSTM2').output
    add1 = Add()([text_output, enc_output])
    fc1 = encoder_decoder.get_layer('fc1')(add1)
    decoder_output = encoder_decoder.get_layer('Output_layer')(fc1)
    decoder_model = Model(inputs=[text_input, enc_output], outputs=decoder_output)
    return encoder_model, decoder_model


def load_chexnet_feature_extractor(weights_path: str) -> Model:
    base = densenet.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3), pooling='avg')
    out = Dense(14, activation='sigmoid', name='predictions')(base.output)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(weights_path)
    # features are the layer before predictions
    feat_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feat_model


def load_image_tensor(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def infer_image_features(chexnet_model: Model, img1_path: str, img2_path: str) -> np.ndarray:
    f1 = chexnet_model.predict(load_image_tensor(img1_path), verbose=0)
    f2 = chexnet_model.predict(load_image_tensor(img2_path), verbose=0)
    return np.concatenate((f1, f2), axis=1)


def _top_k_logits(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= probs.size:
        return probs
    idx = np.argpartition(probs, -k)[-k:]
    masked = np.zeros_like(probs)
    masked[idx] = probs[idx]
    return masked


def _sample_from_probs(probs: np.ndarray, temperature: float = 1.0, top_k: int = 0) -> int:
    p = probs.astype('float64')
    p = np.maximum(p, 1e-9)
    if temperature and temperature != 1.0:
        p = np.power(p, 1.0 / temperature)
    if top_k and top_k > 0:
        p = _top_k_logits(p, top_k)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


def build_tokenizer_from_csv(csv_path: str) -> Tokenizer:
    df = pd.read_csv(csv_path, encoding_errors='ignore')
    # pick report column
    col = next((c for c in df.columns if c.lower() in ['report', 'caption', 'findings', 'impression', 'text']), None)
    if col is None:
        raise ValueError('Could not find a text column (Report/Findings/Impression) in CSV')
    texts = df[col].astype(str).tolist()
    tok = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
    tok.fit_on_texts(texts)
    return tok


def generate_report(encoder_model: tf.keras.Model, decoder_model: tf.keras.Model, tokenizer: Tokenizer,
                    image_features: np.ndarray, max_len: int = 153, top_k: int = 5, temperature: float = 0.8) -> str:
    end_id = tokenizer.word_index.get('endseq')
    start_id = tokenizer.word_index.get('startseq')
    if end_id is None or start_id is None:
        raise RuntimeError('startseq/endseq not in tokenizer vocabulary.')

    enc_feat = encoder_model.predict(image_features, verbose=0)
    seq = [start_id]
    words: List[str] = []
    for _ in range(max_len):
        inp = pad_sequences([seq], max_len, padding='post')
        preds = decoder_model.predict([inp, enc_feat], verbose=0)
        next_id = _sample_from_probs(preds[0], temperature=temperature, top_k=top_k)
        if next_id == end_id or next_id == 0 or next_id not in tokenizer.index_word:
            break
        words.append(tokenizer.index_word[next_id])
        seq.append(next_id)
    return ' '.join(words)


def main():
    ap = argparse.ArgumentParser(description='Terminal inference for report generation (no notebook).')
    ap.add_argument('--csv', required=True, help='CSV containing Person_id / Image1 / Image2 / Report')
    ap.add_argument('--weights', required=True, help='Path to encoder_decoder weights .h5 (e.g., encoder_decoder_epoch_5.weights.h5)')
    ap.add_argument('--chexnet-weights', default='brucechou1983_CheXNet_Keras_0.3.0_weights.h5', help='CheXNet weights .h5')
    ap.add_argument('--vocab-size', type=int, default=1450, help='Vocabulary size used during training (default 1450)')
    ap.add_argument('--max-len', type=int, default=153, help='Max sequence length (default 153)')
    ap.add_argument('--n', type=int, default=5, help='Number of rows to generate')
    ap.add_argument('--top-k', type=int, default=5, help='Top-k sampling (default 5)')
    ap.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (default 0.8)')
    ap.add_argument('--tokenizer-pkl', default=None, help='Path to tokenizer.pkl (recommended). If not provided, tokenizer will be rebuilt from CSV')
    ap.add_argument('--id-col', default=None, help='ID column name (default: Person_id)')
    ap.add_argument('--img1-col', default=None, help='Image1 column name (default: Image1)')
    ap.add_argument('--img2-col', default=None, help='Image2 column name (default: Image2)')
    args = ap.parse_args()

    # Build models and load weights
    print('Building encoder-decoder...')
    encoder_decoder = build_encoder_decoder(args.vocab_size, max_len=args.max_len)
    encoder_decoder.load_weights(args.weights)
    encoder_model, decoder_model = build_encoder_decoder_inference_parts(encoder_decoder)

    if args.tokenizer_pkl and os.path.exists(args.tokenizer_pkl):
        print(f'Loading tokenizer from {args.tokenizer_pkl} ...')
        with open(args.tokenizer_pkl, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        print('Building tokenizer from CSV (warning: may not exactly match training mapping)...')
        tokenizer = build_tokenizer_from_csv(args.csv)

    print('Loading CheXNet feature extractor...')
    chexnet_model = load_chexnet_feature_extractor(args.chexnet_weights)

    df = pd.read_csv(args.csv, encoding_errors='ignore')
    # resolve column names
    id_col = args.id_col or next((c for c in df.columns if c.lower() == 'person_id'), df.columns[0])
    img1_col = args.img1_col or next((c for c in df.columns if c.lower() == 'image1'), None)
    img2_col = args.img2_col or next((c for c in df.columns if c.lower() == 'image2'), None)
    if not img1_col or not img2_col:
        raise ValueError('CSV must contain Image1 and Image2 columns')

    for _, row in df.head(args.n).iterrows():
        pid = str(row[id_col])
        img1 = str(row[img1_col])
        img2 = str(row[img2_col])
        feats = infer_image_features(chexnet_model, img1, img2)
        report = generate_report(encoder_model, decoder_model, tokenizer, feats, max_len=args.max_len, top_k=args.top_k, temperature=args.temperature)
        print('------------------------------------------------------------------------------------------------------')
        print(f'ID: {pid}')
        print(f'Predicted Report: {report}')


if __name__ == '__main__':
    main()


