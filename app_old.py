import os
import pickle
import tempfile
from pathlib import Path
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.sequence import pad_sequences

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


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


def format_report_text(raw_report: str) -> str:
    """Format the generated report into proper medical sections."""
    # Split by common section headers
    sections = {}
    current_section = "findings"
    current_text = []
    
    words = raw_report.split()
    for word in words:
        word_lower = word.lower()
        if word_lower in ['impression:', 'impression', 'impressions:']:
            if current_text:
                sections[current_section] = ' '.join(current_text)
            current_section = "impression"
            current_text = []
        elif word_lower in ['findings:', 'findings']:
            if current_text:
                sections[current_section] = ' '.join(current_text)
            current_section = "findings"
            current_text = []
        else:
            current_text.append(word)
    
    if current_text:
        sections[current_section] = ' '.join(current_text)
    
    # Format output
    formatted = ""
    if "findings" in sections and sections["findings"]:
        formatted += f"**FINDINGS:**\n\n{sections['findings']}\n\n"
    if "impression" in sections and sections["impression"]:
        formatted += f"**IMPRESSION:**\n\n{sections['impression']}"
    
    if not formatted:
        formatted = f"**FINDINGS:**\n\n{raw_report}"
    
    return formatted


def generate_pdf_report(patient_name: str, patient_id: str, age: str, gender: str, 
                       exam_date: str, report_text: str, img1_path: str = None, 
                       img2_path: str = None, logo_path: str = None) -> BytesIO:
    """Generate a professional PDF radiology report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # Container for elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subheader_style = ParagraphStyle(
        'CustomSubHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    label_style = ParagraphStyle(
        'Label',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        fontName='Helvetica-Bold'
    )
    
    value_style = ParagraphStyle(
        'Value',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=colors.HexColor('#1f2937')
    )
    
    # Add logo if available
    if logo_path and os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=1.5*inch, height=1.5*inch)
            logo.hAlign = 'CENTER'
            elements.append(logo)
            elements.append(Spacer(1, 12))
        except:
            pass
    
    # Header
    header_text = "RADIOLOGY REPORT"
    elements.append(Paragraph(header_text, header_style))
    elements.append(Spacer(1, 6))
    
    # AI-Generated Notice
    notice_style = ParagraphStyle(
        'Notice',
        parent=styles['BodyText'],
        fontSize=9,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_CENTER,
        fontStyle='italic'
    )
    elements.append(Paragraph("AI-Generated Report | For Research & Educational Purposes Only", notice_style))
    elements.append(Spacer(1, 20))
    
    # Patient Information Table
    patient_data = [
        [Paragraph('<b>Patient Name:</b>', label_style), Paragraph(patient_name, value_style),
         Paragraph('<b>Patient ID:</b>', label_style), Paragraph(patient_id, value_style)],
        [Paragraph('<b>Age:</b>', label_style), Paragraph(age, value_style),
         Paragraph('<b>Gender:</b>', label_style), Paragraph(gender, value_style)],
        [Paragraph('<b>Exam Date:</b>', label_style), Paragraph(exam_date, value_style),
         Paragraph('<b>Report Date:</b>', label_style), Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M'), value_style)]
    ]
    
    patient_table = Table(patient_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f4f6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Exam Information
    exam_header = Paragraph("EXAMINATION", subheader_style)
    elements.append(exam_header)
    exam_text = Paragraph("Chest X-Ray: Frontal and Lateral Views", body_style)
    elements.append(exam_text)
    elements.append(Spacer(1, 20))
    
    # Report Body
    report_lines = report_text.split('\n')
    for line in report_lines:
        if line.strip():
            if line.strip().startswith('**') and line.strip().endswith('**'):
                # Section header
                clean_line = line.strip().replace('**', '')
                elements.append(Paragraph(clean_line, subheader_style))
            else:
                # Regular text
                clean_line = line.strip().replace('**', '<b>').replace('**', '</b>')
                elements.append(Paragraph(clean_line, body_style))
        else:
            elements.append(Spacer(1, 8))
    
    elements.append(Spacer(1, 30))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['BodyText'],
        fontSize=8,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_CENTER
    )
    
    divider_style = ParagraphStyle(
        'Divider',
        parent=styles['BodyText'],
        fontSize=8,
        textColor=colors.HexColor('#d1d5db'),
        alignment=TA_CENTER
    )
    
    elements.append(Paragraph("_" * 100, divider_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("This report was generated using AI-based image analysis for research purposes.", footer_style))
    elements.append(Paragraph("CheXNet + LSTM Encoder-Decoder Model | Deep Learning in Medical Imaging", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(
        page_title='AI Radiology Report Generator',
        page_icon='🏥',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Custom CSS for professional styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
        }
        .patient-card {
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .report-card {
            background: white;
            border: 2px solid #cbd5e1;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.4);
        }
        .sidebar .sidebar-content {
            background: #f1f5f9;
        }
        div[data-testid="stImage"] {
            border: 2px solid #cbd5e1;
            border-radius: 8px;
            padding: 0.5rem;
            background: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    logo_path = 'logo.png'
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo_path, use_container_width=True)
    
    st.markdown("""
        <div class="main-header">
            <h1>🏥 AI-Powered Radiology Report Generator</h1>
            <p>Automated Chest X-Ray Analysis using Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        st.markdown("---")
        
        weights_h5 = st.text_input(
            '🧠 Decoder Weights',
            'encoder_decoder_epoch_5.weights.h5',
            help='Path to trained encoder-decoder model weights'
        )
        tokenizer_pkl = st.text_input(
            '📝 Tokenizer',
            'models/tokenizer.pkl',
            help='Path to tokenizer pickle file'
        )
        chexnet_h5 = st.text_input(
            '🔬 CheXNet Weights',
            'brucechou1983_CheXNet_Keras_0.3.0_weights.h5',
            help='Path to CheXNet feature extractor weights'
        )
        
        st.markdown("---")
        st.markdown("### 🎛️ Generation Parameters")
        
        top_k = st.slider(
            'Top-k Sampling',
            min_value=1,
            max_value=10,
            value=5,
            help='Number of top candidates to sample from'
        )
        temperature = st.slider(
            'Temperature',
            min_value=0.5,
            max_value=1.5,
            value=0.8,
            step=0.05,
            help='Controls randomness in generation (lower = more conservative)'
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **CheXNet + LSTM Encoder-Decoder**
        
        This system uses:
        - DenseNet121 (CheXNet) for feature extraction
        - LSTM Decoder with GloVe embeddings
        - Trained on IU X-Ray Dataset
        
        *For research and educational purposes.*
        """)

    # Check if required files exist
    required_artifacts = [weights_h5, tokenizer_pkl, chexnet_h5]
    if not all(path and os.path.exists(path) for path in required_artifacts):
        st.error('⚠️ Please ensure all model files are available and paths are correct.')
        st.stop()
    
    # Patient Information Section
    st.markdown("### 👤 Patient Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        patient_name = st.text_input('Patient Name', 'John Doe', key='patient_name')
    with col2:
        patient_id = st.text_input('Patient ID', 'P-' + datetime.now().strftime('%Y%m%d-%H%M'), key='patient_id')
    with col3:
        age = st.text_input('Age', '45', key='age')
    with col4:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'], key='gender')
    
    exam_date = st.date_input('Examination Date', datetime.now(), key='exam_date')
    
    st.markdown("---")
    
    # Image Upload Section
    st.markdown("### 📸 Upload Chest X-Ray Images")
    st.caption('Upload frontal and lateral view chest X-rays (PNG/JPG format)')
    
    uploaded_files = st.file_uploader(
        'Choose X-ray images',
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='upload_images',
        help='Upload 1-2 chest X-ray images. If only one is provided, it will be used for both views.'
    )
    
    uploaded_pair = []
    if uploaded_files:
        uploaded_pair = [{'name': f.name, 'data': f.getvalue()} for f in uploaded_files[:2]]
        if len(uploaded_pair) == 1:
            uploaded_pair *= 2
    
    # Display uploaded images
    if uploaded_pair:
        st.markdown("#### 📋 Uploaded Images")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Frontal View**")
            st.image(uploaded_pair[0]['data'], use_container_width=True, caption=uploaded_pair[0]['name'])
        
        with col2:
            st.markdown("**Lateral View**")
            st.image(uploaded_pair[1]['data'], use_container_width=True, caption=uploaded_pair[1]['name'])
    else:
        st.info('📤 Please upload chest X-ray images to begin analysis')
    
    st.markdown("---")
    
    # Load models
    with st.spinner('🔄 Loading AI models...'):
        encoder_model, decoder_model, tokenizer = load_models(weights_h5, tokenizer_pkl)
        chexnet = load_chexnet(chexnet_h5)
    
    # Generate Report Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button('🚀 Generate Radiology Report', use_container_width=True)
    
    if generate_button:
        if not uploaded_pair:
            st.error('❌ Please upload at least one chest X-ray image before generating a report.')
            st.stop()
        
        # Generate report with progress
        with st.spinner('🔬 Analyzing images and generating report...'):
            temp_paths = []
            try:
                # Save temporary files
                for item in uploaded_pair[:2]:
                    suffix = Path(item['name']).suffix or '.png'
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(item['data'])
                    tmp.flush()
                    tmp.close()
                    temp_paths.append(tmp.name)
                
                # Extract features and generate report
                feats = infer_features(chexnet, temp_paths[0], temp_paths[1])
                raw_report = generate_report(encoder_model, decoder_model, tokenizer, feats, 
                                            top_k=top_k, temperature=temperature)
                formatted_report = format_report_text(raw_report)
                
            finally:
                # Clean up temporary files
                for path in temp_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
        
        # Display Report
        st.success('✅ Report Generated Successfully!')
        st.markdown("---")
        
        st.markdown("""
            <div class="report-card">
                <h2 style="color: #1e3a8a; text-align: center; margin-bottom: 1.5rem;">
                    📋 RADIOLOGY REPORT
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Patient Info Display
        st.markdown("#### Patient Information")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        info_col1.metric("Patient Name", patient_name)
        info_col2.metric("Patient ID", patient_id)
        info_col3.metric("Age", age)
        info_col4.metric("Gender", gender)
        
        st.markdown("#### Examination Details")
        exam_col1, exam_col2 = st.columns(2)
        exam_col1.metric("Exam Date", exam_date.strftime('%Y-%m-%d'))
        exam_col2.metric("Report Generated", datetime.now().strftime('%Y-%m-%d %H:%M'))
        
        st.markdown("---")
        
        # Report Content
        st.markdown("#### 📄 Report Content")
        # Prepare HTML-formatted report
        html_report = formatted_report.replace('**', '<strong>').replace('</strong><strong>', '').replace('\n', '<br>')
        st.markdown(f"""
            <div class="report-card">
                <p><strong>EXAMINATION:</strong> Chest X-Ray - Frontal and Lateral Views</p>
                <br>
                {html_report}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Download PDF Button
        if PDF_AVAILABLE:
            try:
                pdf_buffer = generate_pdf_report(
                    patient_name=patient_name,
                    patient_id=patient_id,
                    age=age,
                    gender=gender,
                    exam_date=exam_date.strftime('%Y-%m-%d'),
                    report_text=formatted_report,
                    logo_path=logo_path if os.path.exists(logo_path) else None
                )
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.download_button(
                        label='📥 Download PDF Report',
                        data=pdf_buffer,
                        file_name=f'radiology_report_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                        mime='application/pdf',
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f'PDF generation encountered an issue: {str(e)}')
        else:
            st.warning('⚠️ Install reportlab to enable PDF download: `pip install reportlab`')
        
        st.markdown("---")
        st.markdown("""
            <p style="text-align: center; color: #6b7280; font-size: 0.9rem;">
                <em>This report was generated using AI-based image analysis for research and educational purposes.</em><br>
                <strong>CheXNet + LSTM Encoder-Decoder Model | Deep Learning in Medical Imaging</strong>
            </p>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()


