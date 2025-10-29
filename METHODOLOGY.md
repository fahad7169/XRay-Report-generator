# Methodology: Automated Chest X-ray Report Generation

## 1. Project Overview

This project implements an automated radiology report generation system for chest X-ray images using deep learning techniques. The system combines computer vision (medical image analysis) with natural language processing to generate textual reports from frontal and lateral chest X-ray images.

### Objectives
- Extract meaningful features from chest X-ray images (both frontal and lateral views)
- Generate coherent and clinically relevant radiology reports
- Automate the report writing process to assist radiologists

---

## 2. Dataset

### 2.1 Primary Datasets
- **NLMCXR Dataset**: NIH Clinical Center chest X-ray dataset containing paired images (frontal and lateral views) with corresponding radiology reports
- **ECGEN-Radiology Dataset**: Additional chest X-ray dataset used for analysis and validation

### 2.2 Data Structure
Each data sample contains:
- **Person_id**: Unique patient identifier
- **Image1**: Path to frontal chest X-ray image (PNG format)
- **Image2**: Path to lateral chest X-ray image (PNG format)
- **Report**: Corresponding radiology report text (Findings and Impressions)

### 2.3 Data Split
- **Training Set**: `Final_Train_Data.csv`
- **Validation Set**: `Final_CV_Data.csv`
- **Test Set**: `Final_Test_Data.csv`

### 2.4 Data Preprocessing
- Images are resized to 224×224 pixels (matching DenseNet121 input requirements)
- Pixel values normalized to [0, 1] range (division by 255.0)
- Text reports are tokenized and processed for sequence-to-sequence learning

---

## 3. Feature Extraction: CheXNet

### 3.1 CheXNet Architecture
CheXNet is a DenseNet121-based convolutional neural network pre-trained for chest X-ray pathology detection:
- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Final Layer**: Dense layer with 14 outputs (14 different pathologies)
- **Input Shape**: (224, 224, 3) RGB images
- **Pooling**: Global Average Pooling

### 3.2 Feature Extraction Process
1. **Image Processing**:
   - Load both frontal (Image1) and lateral (Image2) X-ray images
   - Resize to 224×224 pixels
   - Normalize pixel values
   - Convert BGR to RGB color space

2. **Feature Extraction**:
   - Extract features from both images independently using CheXNet
   - Each image produces a 1024-dimensional feature vector (from the second-to-last layer)
   - Concatenate both feature vectors to create a 2048-dimensional combined feature vector

3. **Implementation**:
   ```python
   # Feature extraction from paired images
   features_frontal = chexnet_model.predict(frontal_image)
   features_lateral = chexnet_model.predict(lateral_image)
   combined_features = concatenate([features_frontal, features_lateral])  # Shape: (2048,)
   ```

---

## 4. Model Architecture: Encoder-Decoder

### 4.1 Overall Architecture
The system uses an encoder-decoder architecture that processes image features and generates text sequences:

```
[Image Features (2048)] → Encoder → [Image Embedding (256)]
                                           ↓
[Text Sequence] → Embedding → LSTM → [Text Embedding (256)]
                                           ↓
                                    [Add Layer]
                                           ↓
                                    [Dense Layers]
                                           ↓
                                    [Next Word Prediction]
```

### 4.2 Encoder Branch (Image Processing)
- **Input**: 2048-dimensional concatenated image features
- **Dense Layer**: 256 units with Glorot uniform initialization (seed=56)
- **Output**: 256-dimensional image embedding vector

### 4.3 Decoder Branch (Text Generation)
- **Input Layers**:
  - **Text Input**: Padded sequences of tokenized words (max length: 153 tokens)
  - **Embedding Layer**: 
    - Pre-trained GloVe embeddings (300 dimensions)
    - Vocabulary size: Variable (typically ~1450 tokens)
    - Uses Out-of-Vocabulary (OOV) token (`<unk>`) for unknown words
    - Embedding matrix is frozen (non-trainable)
  - **LSTM Layers**:
    - **LSTM1**: 256 units, returns sequences (to feed to LSTM2)
    - **LSTM2**: 256 units, returns final hidden state
  - **Dropout**: 0.4 (after LSTM2)

### 4.4 Fusion Layer
- **Add Operation**: Element-wise addition of image embedding and text embedding
- **Dense Layer (fc1)**: 256 units with ReLU activation
- **Dropout**: 0.4
- **Output Layer**: Dense layer with vocabulary size units, softmax activation

### 4.5 Model Specifications
- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss Function**: Sparse Categorical Crossentropy
- **Max Sequence Length**: 153 tokens
- **Batch Size**: 256 (after sequence expansion)

---

## 5. Training Methodology

### 5.1 Text Preprocessing
1. **Tokenization**:
   - Tokenizer configured with OOV token (`<unk>`)
   - Filters special characters: `!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n`
   - Special tokens: `startseq` (start of sequence), `endseq` (end of sequence)

2. **Sequence Preparation**:
   - Reports are tokenized and converted to integer sequences
   - Sequences are padded to max length (153) with post-padding
   - Special start/end tokens are added

### 5.2 Training Data Expansion
For each report, the training pairs are created using a sliding window approach:
- **Input Sequence**: `[startseq, w1, w2, ..., wi]`
- **Target Token**: `wi+1`
- This creates multiple training examples from a single report

**Example**:
- Original report: "The heart is normal in size"
- Training pairs:
  - Input: `[startseq]` → Target: `The`
  - Input: `[startseq, The]` → Target: `heart`
  - Input: `[startseq, The, heart]` → Target: `is`
  - ... and so on

### 5.3 Training Process
1. **Input Features**: Pre-computed image features (2048-dim) for each training sample
2. **Text Sequences**: Padded token sequences
3. **Training Loop**:
   ```python
   for epoch in range(num_epochs):
       model.fit(
           [X_img, X_txt], 
           y_target,
           validation_data=([X_img_val, X_txt_val], y_val),
           batch_size=256,
           shuffle=True
       )
       # Save weights after each epoch
       model.save_weights(f'encoder_decoder_epoch_{epoch+1}.weights.h5')
   ```

### 5.4 Hyperparameters
- **Epochs**: Configurable (typically 20 epochs)
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Dropout Rate**: 0.4
- **Max Sequence Length**: 153
- **Vocabulary Size**: ~1450 tokens (varies based on training data)

---

## 6. Inference and Report Generation

### 6.1 Inference Architecture
During inference, the model is split into two separate models:
1. **Encoder Model**: Processes image features → 256-dim embedding
2. **Decoder Model**: Takes text sequence + image embedding → next word prediction

### 6.2 Text Generation Process
1. **Feature Extraction**:
   - Load frontal and lateral X-ray images
   - Extract features using CheXNet
   - Concatenate features (2048-dim)
   - Encode to 256-dim embedding

2. **Autoregressive Generation**:
   ```
   Start with: [startseq]
   For each time step (up to max_len=153):
     a. Pad current sequence to max_len
     b. Predict next word probability distribution
     c. Sample next word using top-k sampling with temperature
     d. Append word to sequence
     e. If word is 'endseq' or empty, stop generation
   Return: Generated report text
   ```

### 6.3 Sampling Strategy
- **Top-k Sampling**: Limits word selection to top-k most likely words
  - Default: k=5
  - Reduces probability of selecting low-probability words
- **Temperature Scaling**: Controls randomness in sampling
  - Default: temperature=0.8
  - Lower temperature → more deterministic (focused on high-probability words)
  - Higher temperature → more diverse (explores more words)

**Implementation**:
```python
def _top_k_logits(probs, k):
    # Zero out all but top-k logits
    idx = np.argpartition(probs, -k)[-k:]
    masked = np.zeros_like(probs)
    masked[idx] = probs[idx]
    return masked

def _sample_from_probs(probs, temperature=0.8, top_k=5):
    p = np.power(probs, 1.0/temperature) if temperature != 1.0 else probs
    if top_k > 0:
        p = _top_k_logits(p, top_k)
    p = p / p.sum()  # Normalize
    return np.random.choice(len(p), p=p)
```

---

## 7. Model Components

### 7.1 Pre-trained Models
- **CheXNet Weights**: `brucechou1983_CheXNet_Keras_0.3.0_weights.h5`
  - Pre-trained on chest X-ray images for pathology detection
  - Used as a feature extractor (frozen, not fine-tuned)

### 7.2 Pre-trained Embeddings
- **GloVe Embeddings**: 300-dimensional word embeddings
  - File: `glove_vectors` (pickle format)
  - Source: GloVe 6B dataset
  - Used for word representation in the embedding layer

### 7.3 Model Artifacts
- **Trained Weights**: `encoder_decoder_epoch_N.weights.h5` (N = epoch number)
- **Tokenizer**: `tokenizer.pkl` (saved tokenizer with vocabulary mapping)

---

## 8. Implementation Details

### 8.1 Training Pipeline (`train_cli.py`)
```
1. Load training and validation CSV files
2. Extract image features using CheXNet (batch processing)
3. Build tokenizer from training reports
4. Create embedding matrix from GloVe vectors
5. Build encoder-decoder model
6. Prepare training pairs (expanded sequences)
7. Train model for specified epochs
8. Save weights after each epoch
```

### 8.2 Inference Pipeline (`infer_cli.py`)
```
1. Load trained model weights and tokenizer
2. Load CheXNet feature extractor
3. For each test sample:
   a. Extract image features (Image1 + Image2)
   b. Generate report using autoregressive decoding
   c. Display generated report
```

### 8.3 Web Interface (`app.py`)
Streamlit-based web application that provides:
- Interactive report generation
- Image visualization (frontal and lateral views)
- Adjustable generation parameters (top-k, temperature)
- Filtering by disease categories (normal vs. disease-like)

---

## 9. Evaluation Metrics

### 9.1 Text Generation Metrics
- **BLEU Score**: Measures n-gram overlap between generated and reference reports
- **ROUGE Score**: Evaluates recall of overlapping units (typically ROUGE-L for longest common subsequence)
- **CIDEr Score**: Consensus-based evaluation metric

### 9.2 Clinical Relevance
- Keyword matching for pathology detection (pneumothorax, effusion, consolidation, etc.)
- Report coherence and readability assessment
- Comparison with ground truth radiology reports

---

## 10. Technical Stack

### 10.1 Deep Learning Framework
- **TensorFlow/Keras**: Model building and training
- **TensorBoard**: Training visualization and monitoring

### 10.2 Image Processing
- **OpenCV (cv2)**: Image loading and preprocessing
- **DenseNet121**: Pre-trained CNN architecture

### 10.3 Natural Language Processing
- **Tokenizer**: Keras text tokenization
- **GloVe Embeddings**: Pre-trained word vectors

### 10.4 Data Processing
- **Pandas**: CSV data manipulation
- **NumPy**: Numerical computations

### 10.5 Web Interface
- **Streamlit**: Interactive web application

---

## 11. Key Design Decisions

### 11.1 Why Paired Images?
- Chest X-ray analysis typically requires both frontal (PA/AP) and lateral views
- Combining features from both views provides more comprehensive information
- Concatenation (2048-dim) preserves information from both perspectives

### 11.2 Why Encoder-Decoder Architecture?
- Natural fit for image-to-text generation tasks
- Allows model to learn mapping from visual features to textual descriptions
- Can handle variable-length output sequences

### 11.3 Why Frozen GloVe Embeddings?
- Pre-trained embeddings capture general language semantics
- Reduces trainable parameters (faster training, less overfitting)
- Medical terminology benefits from general language understanding

### 11.4 Why Top-k Sampling with Temperature?
- Balances between deterministic (greedy) and random sampling
- Prevents repetitive or generic outputs
- Allows control over generation diversity vs. quality

---

## 12. Limitations and Future Improvements

### 12.1 Current Limitations
- Vocabulary size is limited by training data
- Model may generate generic or repetitive phrases
- Clinical accuracy requires expert validation
- Limited to chest X-ray domain

### 12.2 Potential Improvements
- **Attention Mechanism**: Add attention to focus on relevant image regions
- **Transformer Architecture**: Replace LSTM with Transformer for better long-range dependencies
- **Disease-Specific Fine-tuning**: Specialize models for specific pathologies
- **Multi-task Learning**: Combine report generation with pathology classification
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Clinical Validation**: Extensive evaluation by radiologists

---

## 13. Usage

### 13.1 Training
```bash
python train_cli.py --train-csv Final_Train_Data.csv \
                    --val-csv Final_CV_Data.csv \
                    --epochs 20 \
                    --batch-size 256
```

### 13.2 Inference
```bash
python infer_cli.py --csv Final_Test_Data.csv \
                    --weights encoder_decoder_epoch_20.weights.h5 \
                    --tokenizer-pkl models_oov/tokenizer.pkl \
                    --n 10
```

### 13.3 Web Interface
```bash
streamlit run app.py
```

---

## 14. Conclusion

This methodology describes a complete pipeline for automated radiology report generation from chest X-ray images. The system leverages pre-trained medical imaging models (CheXNet) combined with sequence-to-sequence learning to generate clinically relevant reports. The encoder-decoder architecture with LSTM and pre-trained word embeddings provides a robust framework for image-to-text translation in the medical domain.

