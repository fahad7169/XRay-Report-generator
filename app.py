import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.cache_resource(show_spinner=False)
def load_chexnet(weights_path: str) -> Model:
    base = densenet.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3), pooling='avg')
    out = Dense(14, activation='sigmoid', name='predictions')(base.output)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(weights_path)
    return Model(inputs=model.input, outputs=model.layers[-2].output)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess(path: str) -> np.ndarray:
    img = load_image(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, 0)


def infer_features(chexnet: Model, img1: str, img2: str) -> np.ndarray:
    f1 = chexnet.predict(preprocess(img1), verbose=0)
    f2 = chexnet.predict(preprocess(img2), verbose=0)
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


def generate_report(encoder_model, decoder_model, tokenizer, feats, max_len=153, top_k=5, temperature=0.8):
    end_id = tokenizer.word_index.get('endseq')
    start_id = tokenizer.word_index.get('startseq')
    if end_id is None or start_id is None:
        return "Tokenizer missing startseq/endseq."
    enc_feat = encoder_model.predict(feats, verbose=0)
    seq = [start_id]
    words = []
    for _ in range(max_len):
        inp = pad_sequences([seq], max_len, padding='post')
        preds = decoder_model.predict([inp, enc_feat], verbose=0)
        nxt = _sample_from_probs(preds[0], temperature=temperature, top_k=top_k)
        if nxt == end_id or nxt == 0 or nxt not in tokenizer.index_word:
            break
        words.append(tokenizer.index_word[nxt])
        seq.append(nxt)
    return ' '.join(words)


@st.cache_resource(show_spinner=False)
def load_models(weights_h5: str, tokenizer_pkl: str):
    # Build same architecture as in training/infer_cli
    from infer_cli import build_encoder_decoder, build_encoder_decoder_inference_parts
    # Need vocab size from tokenizer
    with open(tokenizer_pkl, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    encoder_decoder = build_encoder_decoder(vocab_size)
    encoder_decoder.load_weights(weights_h5)
    encoder_model, decoder_model = build_encoder_decoder_inference_parts(encoder_decoder)
    return encoder_model, decoder_model, tokenizer


def main():
    st.set_page_config(page_title='Chest X-ray Report Generator', layout='wide')
    st.title('Chest X-ray Report Generator (Demo)')

    with st.sidebar:
        st.header('Configuration')
        csv_path = st.text_input('CSV path', 'Final_CV_Data.csv')
        id_col = st.text_input('ID column', 'Person_id')
        img1_col = st.text_input('Image1 column', 'Image1')
        img2_col = st.text_input('Image2 column', 'Image2')
        report_col = st.text_input('Text column (for filtering)', 'Report')
        weights_h5 = st.text_input('Decoder weights (.h5)', 'encoder_decoder_epoch_5.weights.h5')
        tokenizer_pkl = st.text_input('Tokenizer .pkl', 'models_oov/tokenizer.pkl')
        chexnet_h5 = st.text_input('CheXNet weights (.h5)', 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
        top_k = st.slider('Top-k', 1, 10, 5)
        temperature = st.slider('Temperature', 0.5, 1.5, 0.8, 0.05)

    if not (os.path.exists(csv_path) and os.path.exists(weights_h5) and os.path.exists(tokenizer_pkl) and os.path.exists(chexnet_h5)):
        st.warning('Please ensure CSV, weights, tokenizer, and CheXNet weights paths are valid.')
        return

    df = pd.read_csv(csv_path, encoding_errors='ignore')
    if not all(c in df.columns for c in [id_col, img1_col, img2_col]):
        st.error('CSV is missing required columns.')
        return

    # Filtering
    st.subheader('Filter')
    disease_keywords = ['pneumothorax','effusion','consolidation','opacity','atelectasis','cardiomegaly','nodule','mass','granuloma']
    kind = st.radio('Subset', ['Disease-like', 'Normal-like'], horizontal=True)
    tx = df.get(report_col, pd.Series([''] * len(df))).astype(str).str.lower()
    mask = tx.apply(lambda t: any(k in t for k in disease_keywords))
    subset = df[mask] if kind == 'Disease-like' else df[~mask]
    st.caption(f'Showing {len(subset)} rows from {kind} subset.')

    # Pick one sample
    idx = st.number_input('Row index', min_value=0, max_value=max(0, len(subset)-1), value=0, step=1)
    if len(subset) == 0:
        st.info('No rows to display.')
        return
    row = subset.iloc[int(idx)]

    c1, c2 = st.columns(2)
    with c1:
        st.image(load_image(str(row[img1_col])), caption=str(row[img1_col]), use_container_width=True)
    with c2:
        st.image(load_image(str(row[img2_col])), caption=str(row[img2_col]), use_container_width=True)

    # Load models
    encoder_model, decoder_model, tokenizer = load_models(weights_h5, tokenizer_pkl)
    chexnet = load_chexnet(chexnet_h5)

    if st.button('Generate Report'):
        feats = infer_features(chexnet, str(row[img1_col]), str(row[img2_col]))
        pred = generate_report(encoder_model, decoder_model, tokenizer, feats, top_k=top_k, temperature=temperature)
        st.success('Predicted Report:')
        st.write(pred)
        if report_col in df.columns:
            st.caption('Reference text (from CSV):')
            st.write(str(row[report_col]))


if __name__ == '__main__':
    main()


