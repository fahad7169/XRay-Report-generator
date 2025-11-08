# 🏥 AI-Powered Radiology Report Generator

Professional medical-grade web application for automated chest X-ray report generation using deep learning.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Logo (Optional)
Place your institutional logo as `logo.png` in the root directory.
- Format: PNG (transparent background recommended)
- Size: 400×400 pixels (square or landscape)

### 3. Run the Application
```bash
streamlit run app.py
```
Or double-click: `run_demo.bat`

The app opens at `http://localhost:8501`

---

## ✨ Features

- 🤖 **AI-Powered**: CheXNet (DenseNet121) + LSTM Encoder-Decoder
- 📋 **Structured Reports**: Automatic FINDINGS and IMPRESSION sections
- 📥 **PDF Export**: Professional, branded downloadable reports
- 👤 **Patient Management**: Complete demographic tracking
- 🎨 **Professional UI**: Medical-grade interface with custom styling

---

## 📊 Architecture

```
Chest X-Ray Images → CheXNet (DenseNet121) → 2048-dim Features
  ↓
Dense Encoder (256-dim) → LSTM Decoder + GloVe → Medical Text
  ↓
Formatted Report + PDF Download
```

**Training Data:** IU X-Ray Dataset (Indiana University)

---

## 🎯 Usage Workflow

1. **Enter Patient Information** - Name, ID, Age, Gender, Exam Date
2. **Upload X-Ray Images** - Frontal and/or lateral views (PNG/JPG)
3. **Generate Report** - AI analyzes images (~5 seconds)
4. **View Structured Report** - FINDINGS and IMPRESSION sections
5. **Download PDF** - Professional report with logo and patient info

---

## 📁 Project Structure

```
my_project/
├── app.py                          # Main Streamlit application
├── train_cli.py                    # Model training
├── infer_cli.py                    # CLI inference
├── logo.png                        # Your institutional logo
├── run_demo.bat                    # Quick launcher
├── requirements.txt                # Dependencies
├── models/
│   └── tokenizer.pkl
├── demo_samples/                   # Sample X-ray images
└── README.md                       # This file
```

---

## 🎤 5-Minute Demo Script

1. **Introduction** (30s) - Show professional interface and logo
2. **Patient Info** (30s) - Fill demographics with auto-generated ID
3. **Upload Images** (45s) - Drag-and-drop chest X-rays
4. **AI Analysis** (1m) - Generate report, explain CheXNet + LSTM
5. **View Report** (1m) - Show structured FINDINGS/IMPRESSION
6. **Download PDF** (45s) - Professional PDF with branding

**Key Talking Points:**
- "Medical-grade AI interface trained on IU X-Ray Dataset"
- "CheXNet DenseNet121 for feature extraction"
- "LSTM encoder-decoder with GloVe embeddings"
- "Structured reports following radiology standards"
- "One-click PDF with institutional branding"

See `DEMO_CHECKLIST.md` for detailed preparation guide.

---

## ⚙️ Configuration

### Model Files Required:
- `encoder_decoder_epoch_5.weights.h5` - Trained model weights
- `brucechou1983_CheXNet_Keras_0.3.0_weights.h5` - CheXNet weights
- `models/tokenizer.pkl` - Tokenizer for text processing

### Adjustable Parameters (in sidebar):
- **Top-k Sampling** (1-10) - Number of top candidates
- **Temperature** (0.5-1.5) - Randomness control

---

## 🛠️ Troubleshooting

**Models don't load**
- Check file paths in sidebar
- Verify all model files exist

**Logo doesn't appear**
- Ensure `logo.png` exists in root directory
- Use PNG or JPG format

**PDF download unavailable**
- Install reportlab: `pip install reportlab`

**Images won't upload**
- Ensure PNG or JPG format
- Check file size (< 10 MB)

---

## 📚 Technical Details

### Model Specifications
- **Feature Extractor**: CheXNet (DenseNet121), 2048-dim output
- **Encoder**: Dense layer (2048 → 256 dimensions)
- **Decoder**: 2× LSTM layers (256 units), GloVe embeddings (300-dim)
- **Training**: IU X-Ray Dataset, teacher forcing, Adam optimizer
- **Inference**: Top-k sampling, temperature-controlled generation

### Performance
- **Speed**: ~5 seconds per report
- **Input**: 224×224 RGB images (2 views)
- **Output**: Structured medical text (FINDINGS + IMPRESSION)

---

## ⚠️ Important Notes

**For Research & Educational Use Only**

This system is designed for:
- ✅ Academic demonstrations
- ✅ Research projects
- ✅ Educational purposes

**NOT intended for:**
- ❌ Clinical diagnosis without validation
- ❌ Medical practice without regulatory approval
- ❌ Patient care decisions

**Disclaimer:** AI-generated reports may contain errors. Clinical deployment requires extensive validation, regulatory approval, and professional oversight.

---

## 📖 Additional Documentation

- **DEMO_CHECKLIST.md** - Complete demonstration preparation guide
- **METHODOLOGY.md** - Technical methodology and architecture
- **PROJECT_REPORT.md** - Full academic project report
- **FLOWCHART_PROMPT.md** - System flowchart generation

---

## 🙏 Acknowledgments

- **CheXNet** - Pre-trained chest X-ray model
- **IU X-Ray Dataset** - Indiana University training data
- **Streamlit** - Web framework
- **TensorFlow/Keras** - Deep learning
- **ReportLab** - PDF generation

---

## 📜 Version

**Version 2.0** - November 2025

Professional medical-grade interface with PDF generation and comprehensive patient workflow.

---

**Ready to impress! 🚀**

For detailed demo preparation, see `DEMO_CHECKLIST.md`

