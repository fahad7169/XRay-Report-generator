import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Embedding, LSTM, Add, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2


def guess_cols(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    id_col = next((c for c in df.columns if c.lower() == 'person_id'), df.columns[0])
    img1_col = next((c for c in df.columns if c.lower() == 'image1'), None)
    img2_col = next((c for c in df.columns if c.lower() == 'image2'), None)
    rep_col = next((c for c in df.columns if c.lower() in ['report', 'findings', 'impression', 'caption', 'text']), None)
    if not img1_col or not img2_col or not rep_col:
        raise ValueError('CSV must contain Image1, Image2 and a report text column (Report/Findings/Impression).')
    return id_col, img1_col, img2_col, rep_col


def load_image_tensor(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def build_chexnet_feature_extractor(weights_path: str) -> Model:
    base = densenet.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3), pooling='avg')
    out = Dense(14, activation='sigmoid', name='predictions')(base.output)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(weights_path)
    feat_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feat_model


def compute_pair_features(chexnet_model: Model, img1_paths: List[str], img2_paths: List[str], batch_size: int = 16) -> np.ndarray:
    feats = []
    n = len(img1_paths)
    for i in range(0, n, batch_size):
        b1 = [load_image_tensor(p) for p in img1_paths[i:i+batch_size]]
        b2 = [load_image_tensor(p) for p in img2_paths[i:i+batch_size]]
        f1 = chexnet_model.predict(np.vstack(b1), verbose=0)
        f2 = chexnet_model.predict(np.vstack(b2), verbose=0)
        feats.append(np.concatenate((f1, f2), axis=1))
    return np.vstack(feats)


def texts_to_tokenizer_oov(texts: List[str]) -> Tokenizer:
    tok = Tokenizer(oov_token='<unk>', filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
    tok.fit_on_texts(texts)
    return tok


def build_embedding_matrix(tokenizer: Tokenizer, glove_pickle_path: str) -> np.ndarray:
    with open(glove_pickle_path, 'rb') as f:
        glove = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    mat = np.zeros((vocab_size, 300), dtype='float32')
    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        vec = glove.get(word)
        if vec is not None:
            mat[idx] = vec
        # else keep zeros (including <unk>)
    return mat


def build_model(vocab_size: int, embedding_matrix: np.ndarray, max_len: int = 153) -> tf.keras.Model:
    # image branch (input is concatenated features of both images: 2048)
    image_input = Input(shape=(2048,), name='Image_input')
    dense_encoder = Dense(256, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=56), name='dense_encoder')(image_input)

    # text branch
    text_input = Input(shape=(max_len,), name='Text_Input')
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=max_len, mask_zero=True,
                                trainable=False, weights=[embedding_matrix], name='Embedding_layer')(text_input)
    lstm1 = LSTM(256, return_sequences=True, name='LSTM1')(embedding_layer)
    lstm2 = LSTM(256, name='LSTM2')(lstm1)
    dropout1 = Dropout(0.4, name='dropout1')(lstm2)

    add = Add(name='add')([dense_encoder, dropout1])
    fc1 = Dense(256, activation='relu', name='fc1')(add)
    dropout2 = Dropout(0.4, name='dropout2')(fc1)
    output = Dense(vocab_size, activation='softmax', name='Output_layer')(dropout2)

    model = Model(inputs=[image_input, text_input], outputs=output, name='encoder_decoder')
    model.compile(optimizer=Adam(1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    return model


def make_training_pairs(features: np.ndarray, reports: List[str], tokenizer: Tokenizer, max_len: int = 153) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_img, X_txt, y = [], [], []
    for i, rep in enumerate(reports):
        seq = [tokenizer.word_index[w] for w in rep.split() if w in tokenizer.word_index]
        for j in range(1, len(seq)):
            in_seq = seq[:j]
            out_tok = seq[j]
            X_img.append(features[i])
            X_txt.append(in_seq)
            y.append(out_tok)
    X_txt = pad_sequences(X_txt, maxlen=max_len, padding='post')
    return np.array(X_img), np.array(X_txt), np.array(y)


def main():
    ap = argparse.ArgumentParser(description='Train decoder with OOV tokenizer and sparse loss (terminal).')
    ap.add_argument('--train-csv', default='Final_Train_Data.csv')
    ap.add_argument('--val-csv', default='Final_CV_Data.csv')
    ap.add_argument('--chexnet-weights', default='brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
    ap.add_argument('--glove-pkl', default='glove_vectors', help='Pickle file with GloVe vectors')
    ap.add_argument('--save-dir', default='models_oov')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--max-len', type=int, default=153)
    ap.add_argument('--batch-size', type=int, default=256, help='Batch size for model.fit after expansion')
    ap.add_argument('--limit-train', type=int, default=0, help='Limit rows for quick test (0 = all)')
    ap.add_argument('--limit-val', type=int, default=0, help='Limit rows for quick test (0 = all)')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # load CSVs
    train_df = pd.read_csv(args.train_csv, encoding_errors='ignore')
    val_df = pd.read_csv(args.val_csv, encoding_errors='ignore')
    id_col, img1_col, img2_col, rep_col = guess_cols(train_df)
    _, _, _, rep_col_val = guess_cols(val_df)

    if args.limit_train:
        train_df = train_df.head(args.limit_train)
    if args.limit_val:
        val_df = val_df.head(args.limit_val)

    # tokenizer with OOV
    train_texts = train_df[rep_col].astype(str).tolist()
    tokenizer = texts_to_tokenizer_oov(train_texts)
    with open(os.path.join(args.save_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    # embeddings
    embedding_matrix = build_embedding_matrix(tokenizer, args.glove_pkl)
    vocab_size = embedding_matrix.shape[0]

    # image features
    chexnet = build_chexnet_feature_extractor(args.chexnet_weights)
    train_feats = compute_pair_features(chexnet, train_df[img1_col].astype(str).tolist(), train_df[img2_col].astype(str).tolist())
    val_feats = compute_pair_features(chexnet, val_df[img1_col].astype(str).tolist(), val_df[img2_col].astype(str).tolist())

    # training pairs (expanded)
    X_img_tr, X_txt_tr, y_tr = make_training_pairs(train_feats, train_df[rep_col].astype(str).tolist(), tokenizer, max_len=args.max_len)
    X_img_val, X_txt_val, y_val = make_training_pairs(val_feats, val_df[rep_col_val].astype(str).tolist(), tokenizer, max_len=args.max_len)

    # model
    model = build_model(vocab_size, embedding_matrix, max_len=args.max_len)

    for epoch in range(args.epochs):
        model.fit([X_img_tr, X_txt_tr], y_tr, validation_data=([X_img_val, X_txt_val], y_val), batch_size=args.batch_size, epochs=1, shuffle=True)
        wpath = os.path.join(args.save_dir, f'encoder_decoder_epoch_{epoch+1}.weights.h5')
        model.save_weights(wpath)
        print(f'Saved weights -> {wpath}')


if __name__ == '__main__':
    main()


