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
    from infer_cli import build_encoder_decoder, build_encoder_decoder_inference_parts
    with open(tokenizer_pkl, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    encoder_decoder = build_encoder_decoder(vocab_size)
    encoder_decoder.load_weights(weights_h5)
    encoder_model, decoder_model = build_encoder_decoder_inference_parts(encoder_decoder)
    return encoder_model, decoder_model, tokenizer


def format_report_text(raw_report: str) -> str:
    """Format the generated report into proper medical sections."""
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
    """Generate a professional, formal PDF radiology report with clear formatting."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=60, leftMargin=60,
                          topMargin=50, bottomMargin=50)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Define professional color scheme
    primary_blue = colors.HexColor('#1e40af')
    secondary_blue = colors.HexColor('#3b82f6')
    dark_gray = colors.HexColor('#1f2937')
    medium_gray = colors.HexColor('#6b7280')
    light_gray = colors.HexColor('#f3f4f6')
    border_gray = colors.HexColor('#e5e7eb')
    
    # Main Title Style - Professional and Bold
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=primary_blue,
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=28
    )
    
    # Subtitle Style
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=medium_gray,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica',
        leading=12
    )
    
    # Section Header Style - Clear and Bold
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=primary_blue,
        spaceAfter=10,
        spaceBefore=16,
        fontName='Helvetica-Bold',
        leading=16,
        borderWidth=0,
        borderColor=primary_blue,
        borderPadding=0,
        leftIndent=0
    )
    
    # Body Text Style - Easy to Read
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=11,
        leading=18,
        alignment=TA_LEFT,
        spaceAfter=10,
        textColor=dark_gray,
        fontName='Helvetica'
    )
    
    # Label and Value Styles
    label_style = ParagraphStyle(
        'Label',
        parent=styles['Normal'],
        fontSize=10,
        textColor=dark_gray,
        fontName='Helvetica-Bold'
    )
    
    value_style = ParagraphStyle(
        'Value',
        parent=styles['Normal'],
        fontSize=10,
        textColor=dark_gray,
        fontName='Helvetica'
    )
    
    # ===== HEADER SECTION =====
    # Logo (if available)
    if logo_path and os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=1.2*inch, height=1.2*inch)
            logo.hAlign = 'CENTER'
            elements.append(logo)
            elements.append(Spacer(1, 10))
        except:
            pass
    
    # Main Title with Top Border
    elements.append(Paragraph("RADIOLOGY REPORT", title_style))
    
    # Subtitle with Classification
    elements.append(Paragraph(
        "Chest Radiography | AI-Assisted Analysis",
        subtitle_style
    ))
    
    # Horizontal line separator
    from reportlab.platypus import HRFlowable
    elements.append(HRFlowable(width="100%", thickness=2, color=primary_blue, 
                               spaceBefore=0, spaceAfter=20))
    
    # ===== PATIENT INFORMATION SECTION =====
    elements.append(Paragraph("PATIENT INFORMATION", section_header_style))
    
    # Patient info table with better styling
    patient_data = [
        [Paragraph('<b>Patient Name:</b>', label_style), Paragraph(patient_name, value_style),
         Paragraph('<b>Patient ID:</b>', label_style), Paragraph(patient_id, value_style)],
        [Paragraph('<b>Age:</b>', label_style), Paragraph(f"{age} years", value_style),
         Paragraph('<b>Gender:</b>', label_style), Paragraph(gender, value_style)],
        [Paragraph('<b>Examination Date:</b>', label_style), Paragraph(exam_date, value_style),
         Paragraph('<b>Report Date:</b>', label_style), 
         Paragraph(datetime.now().strftime('%Y-%m-%d'), value_style)]
    ]
    
    patient_table = Table(patient_data, colWidths=[1.3*inch, 2*inch, 1.3*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_gray),
        ('GRID', (0, 0), (-1, -1), 0.5, border_gray),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # ===== REPORT CONTENT SECTION =====
    # Parse and format the report text properly
    report_sections = {}
    current_section = None
    
    # Ensure we have report text
    if not report_text or not report_text.strip():
        report_text = "No findings generated. Please ensure the images are properly formatted chest X-rays."
    
    lines = report_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a section header
        if line.startswith('**') and line.endswith('**'):
            section_name = line.replace('**', '').strip().upper()
            # Remove trailing colons from section names
            section_name = section_name.rstrip(':')
            current_section = section_name
            report_sections[current_section] = []
        elif current_section:
            report_sections[current_section].append(line)
        else:
            # If no section has been defined yet, add to a default section
            if 'CONTENT' not in report_sections:
                report_sections['CONTENT'] = []
            report_sections['CONTENT'].append(line)
    
    # Helper function to clean and escape text for PDF
    def clean_for_pdf(text):
        """Clean text for PDF rendering"""
        # Remove markdown formatting
        text = text.replace('**', '')
        # Escape special XML/HTML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text.strip()
    
    # Display FINDINGS section
    if 'FINDINGS' in report_sections:
        elements.append(Paragraph("FINDINGS", section_header_style))
        findings_text = ' '.join(report_sections['FINDINGS'])
        findings_text = clean_for_pdf(findings_text)
        if findings_text:
            findings_para = Paragraph(findings_text, body_style)
            elements.append(findings_para)
            elements.append(Spacer(1, 15))
    elif 'CONTENT' in report_sections:
        # If no explicit FINDINGS section, use CONTENT
        elements.append(Paragraph("FINDINGS", section_header_style))
        findings_text = ' '.join(report_sections['CONTENT'])
        findings_text = clean_for_pdf(findings_text)
        if findings_text:
            findings_para = Paragraph(findings_text, body_style)
            elements.append(findings_para)
            elements.append(Spacer(1, 15))
    else:
        # Fallback: display the entire report as findings
        elements.append(Paragraph("FINDINGS", section_header_style))
        # Remove markdown formatting
        clean_text = report_text.replace('**FINDINGS:**', '').replace('**IMPRESSION:**', '').strip()
        clean_text = clean_for_pdf(clean_text)
        if clean_text:
            # Split into paragraphs if it's too long
            paragraphs = clean_text.split('. ')
            full_text = '. '.join(paragraphs)
            findings_para = Paragraph(full_text, body_style)
            elements.append(findings_para)
            elements.append(Spacer(1, 15))
    
    # Display IMPRESSION section
    if 'IMPRESSION' in report_sections:
        elements.append(Paragraph("IMPRESSION", section_header_style))
        impression_text = ' '.join(report_sections['IMPRESSION'])
        impression_text = clean_for_pdf(impression_text)
        if impression_text:
            impression_para = Paragraph(impression_text, body_style)
            elements.append(impression_para)
            elements.append(Spacer(1, 20))
    
    # ===== DISCLAIMER SECTION =====
    elements.append(HRFlowable(width="100%", thickness=1, color=border_gray, 
                               spaceBefore=20, spaceAfter=15))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=medium_gray,
        alignment=TA_JUSTIFY,
        leading=13,
        spaceAfter=8
    )
    
    disclaimer_title_style = ParagraphStyle(
        'DisclaimerTitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=dark_gray,
        fontName='Helvetica-Bold',
        spaceAfter=6
    )
    
    elements.append(Paragraph("IMPORTANT DISCLAIMER", disclaimer_title_style))
    elements.append(Paragraph(
        "This report was generated using artificial intelligence and deep learning algorithms "
        "for research and educational purposes only. This AI-generated report should not be "
        "used as the sole basis for clinical decision-making. All findings should be verified "
        "by a qualified radiologist or healthcare professional.",
        disclaimer_style
    ))
    
    # ===== FOOTER SECTION =====
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=medium_gray,
        alignment=TA_CENTER,
        leading=11
    )
    
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=border_gray, 
                               spaceBefore=5, spaceAfter=10))
    
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M')} | "
        "AI Model: CheXNet + LSTM Encoder-Decoder",
        footer_style
    ))
    elements.append(Paragraph(
        "© 2024 AI Medical Imaging Research | For Research & Educational Use Only",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def load_custom_css():
    """Load premium, elegant medical-grade UI"""
    st.markdown("""
        <style>
        /* Import Premium Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main Container */
        .main .block-container {
            padding: 3rem 2rem;
            max-width: 1600px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        /* Elegant Background with Subtle Pattern */
        .stApp {
            background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f8fafc 100%);
            background-attachment: fixed;
        }
        
        /* Premium Content Cards */
        .main-card {
            background: white;
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06), 0 2px 8px rgba(0, 0, 0, 0.03);
            margin-bottom: 2rem;
            border: 1px solid #e2e8f0;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .main-card:hover {
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.04);
            transform: translateY(-2px);
        }
        
        /* Hero Header - Medical Grade */
        .hero-header {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-radius: 24px;
            padding: 4rem 3rem;
            text-align: center;
            margin-bottom: 3rem;
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.08);
            position: relative;
        }
        
        .hero-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        }
        
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
            color: #64748b;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        
        /* Section Headers */
        .section-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            color: #1e293b !important;
            margin-bottom: 1.5rem !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.75rem !important;
            padding-bottom: 0.75rem !important;
            border-bottom: 2px solid #e2e8f0 !important;
            background: transparent !important;
        }
        
        /* Fix for section title text */
        .section-title * {
            color: #1e293b !important;
        }
        
        /* Premium Input Fields */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stDateInput > div > div > input {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 0.875rem 1.25rem;
            font-size: 0.95rem;
            background: #f8fafc;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: #1e293b;
            font-weight: 500;
        }
        
        .stTextInput > div > div > input:hover,
        .stSelectbox > div > div > select:hover,
        .stDateInput > div > div > input:hover {
            border-color: #cbd5e1;
            background: #ffffff;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stDateInput > div > div > input:focus {
            border-color: #3b82f6;
            background: white;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08);
            outline: none;
        }
        
        /* Select Box Arrow */
        .stSelectbox > div > div > svg {
            color: #3b82f6;
        }
        
        /* Labels */
        .stTextInput label,
        .stSelectbox label,
        .stDateInput label {
            font-weight: 600 !important;
            color: #475569 !important;
            font-size: 0.875rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        /* Ensure all paragraph text is visible */
        p, .stMarkdown p {
            color: #64748b !important;
        }
        
        /* Heading text */
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b !important;
        }
        
        /* Premium Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 1rem 2.5rem !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3), 0 2px 8px rgba(59, 130, 246, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
            letter-spacing: 0.02em;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4), 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }
        
        .stButton > button p {
            color: white !important;
            margin: 0;
        }
        
        /* Download Button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 1rem 2.5rem;
            font-size: 1.05rem;
            font-weight: 600;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3), 0 2px 8px rgba(16, 185, 129, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
            letter-spacing: 0.02em;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4), 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* File Uploader - Enhanced Visibility */
        .stFileUploader {
            border: 3px dashed #94a3b8 !important;
            border-radius: 16px !important;
            padding: 2.5rem 2rem !important;
            background: #ffffff !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            position: relative;
        }
        
        .stFileUploader:hover {
            border-color: #3b82f6 !important;
            background: #f8fafc !important;
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12);
            transform: translateY(-2px);
        }
        
        .stFileUploader label {
            font-weight: 600 !important;
            color: #1e293b !important;
            font-size: 1rem !important;
        }
        
        .stFileUploader section {
            background: transparent !important;
        }
        
        .stFileUploader section > div {
            background: transparent !important;
        }
        
        /* File uploader text visibility - Comprehensive Fix */
        .stFileUploader small {
            color: #64748b !important;
            font-size: 0.875rem !important;
        }
        
        .stFileUploader [data-testid="stMarkdownContainer"] {
            color: #1e293b !important;
        }
        
        .stFileUploader [data-testid="stMarkdownContainer"] p {
            color: #64748b !important;
        }
        
        /* Fix all text elements in file uploader */
        .stFileUploader * {
            color: #1e293b !important;
        }
        
        .stFileUploader p,
        .stFileUploader span,
        .stFileUploader div,
        .stFileUploader label {
            color: #1e293b !important;
        }
        
        /* File list items */
        .stFileUploader [data-testid="stFileUploader"] * {
            color: #1e293b !important;
        }
        
        /* File name and size text */
        .stFileUploader [data-testid="stFileUploaderFileName"],
        .stFileUploader [data-testid="stFileUploaderFileSize"] {
            color: #475569 !important;
        }
        
        /* Drag and drop area text */
        .stFileUploader [role="button"] {
            color: #1e293b !important;
        }
        
        .stFileUploader [role="button"] p,
        .stFileUploader [role="button"] span,
        .stFileUploader [role="button"] div {
            color: #1e293b !important;
        }
        
        /* Browse files button - ensure text is visible */
        .stFileUploader button {
            background: #1e293b !important;
            color: white !important;
            border: 1px solid #1e293b !important;
        }
        
        .stFileUploader button:hover {
            background: #3b82f6 !important;
            border-color: #3b82f6 !important;
        }
        
        .stFileUploader button p,
        .stFileUploader button span,
        .stFileUploader button div {
            color: white !important;
        }
        
        /* Baseweb button styling */
        .stFileUploader [data-baseweb="button"] {
            background: #1e293b !important;
            color: white !important;
        }
        
        .stFileUploader [data-baseweb="button"] * {
            color: white !important;
        }
        
        /* Remove file X button - keep it visible */
        .stFileUploader [data-testid="stFileUploaderDeleteButton"] {
            color: #64748b !important;
        }
        
        /* Override any white text */
        .stFileUploader *[style*="color: white"],
        .stFileUploader *[style*="color: #fff"],
        .stFileUploader *[style*="color: #ffffff"] {
            color: #1e293b !important;
        }
        
        /* Image Display */
        div[data-testid="stImage"] {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            background: white;
            padding: 0.5rem;
            position: relative;
        }
        
        div[data-testid="stImage"]:hover {
            transform: scale(1.03);
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
        }
        
        div[data-testid="stImage"] img {
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stImage"]:hover img {
            filter: brightness(1.05);
        }
        
        /* Report Card */
        .report-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 3rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            border-left: 6px solid #3b82f6;
            border-top: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .report-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            padding-bottom: 1rem;
        }
        
        .report-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 2px;
        }
        
        .report-section {
            margin-bottom: 2rem;
            line-height: 1.8;
        }
        
        .report-section strong {
            color: #1e40af;
            font-size: 1.15rem;
            display: block;
            margin-bottom: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        
        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1e40af;
            font-family: 'Poppins', sans-serif;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 1.25rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            background: white;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
            transform: translateY(-2px);
        }
        
        /* Sidebar - Medical Professional */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e40af 0%, #1e3a8a 50%, #1e293b 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: rgba(255, 255, 255, 0.95);
        }
        
        section[data-testid="stSidebar"] h3 {
            color: white;
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 0.02em;
        }
        
        section[data-testid="stSidebar"] .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            font-weight: 500;
        }
        
        section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
            background: rgba(255, 255, 255, 0.18);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        section[data-testid="stSidebar"] .stTextInput label {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 600;
        }
        
        /* Sidebar Slider */
        section[data-testid="stSidebar"] .stSlider label {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 600;
        }
        
        /* Alert Styles - Enhanced */
        .stSuccess, .stError, .stInfo, .stWarning {
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            border: none;
            font-weight: 500;
            backdrop-filter: blur(10px);
            animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
        }
        
        .stError {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            box-shadow: 0 4px 16px rgba(239, 68, 68, 0.3);
        }
        
        .stInfo {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        }
        
        .stWarning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            box-shadow: 0 4px 16px rgba(245, 158, 11, 0.3);
        }
        
        /* Alert Icons */
        .stSuccess::before,
        .stError::before,
        .stInfo::before,
        .stWarning::before {
            margin-right: 0.5rem;
            font-weight: bold;
        }
        
        /* Loading Spinner - Enhanced and Fixed */
        .stSpinner > div {
            border-top-color: #3b82f6 !important;
            border-right-color: #3b82f6 !important;
        }
        
        /* Spinner text */
        .stSpinner > div + div,
        .stSpinner p {
            color: #1e40af !important;
            font-weight: 600 !important;
            margin-top: 1rem;
        }
        
        /* Alert text visibility */
        .stAlert p {
            color: inherit !important;
            margin: 0;
        }
        
        /* Divider */
        hr {
            margin: 2.5rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        }
        
        /* Smooth Animations - Fixed */
        @keyframes fadeInUp {
            0% { 
                opacity: 0; 
                transform: translateY(20px);
            }
            100% { 
                opacity: 1; 
                transform: translateY(0);
            }
        }
        
        @keyframes slideIn {
            0% {
                opacity: 0;
                transform: translateX(-10px);
            }
            100% {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: 0.9;
                transform: scale(1.02);
            }
        }
        
        /* Apply animations properly */
        .hero-header {
            animation: fadeInUp 0.6s ease-out forwards;
        }
        
        /* Ensure elements are visible during animation */
        [data-testid="stMarkdownContainer"] p {
            color: #64748b !important;
        }
        
        /* Premium Scrollbar */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 10px;
            border: 2px solid #f1f5f9;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        }
        
        /* Responsive Typography */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
            
            .hero-subtitle {
                font-size: 0.95rem;
            }
            
            .main-card {
                padding: 1.5rem;
            }
        }
        
        /* Global text visibility fixes */
        * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* Ensure all text is visible */
        .stMarkdown, .stText {
            color: #1e293b;
        }
        
        .stMarkdown p, .stText p {
            color: #64748b !important;
        }
        
        /* Fix for file uploader drag area text */
        .stFileUploader [role="button"] {
            color: #1e293b !important;
        }
        
        /* Specific fix for drag and drop text */
        .stFileUploader [role="button"] * {
            color: #1e293b !important;
        }
        
        /* Override Streamlit's default white text in uploader */
        .stFileUploader [data-baseweb="base-input"] {
            color: #1e293b !important;
        }
        
        /* File uploader status messages */
        .stFileUploader [data-testid="stFileUploaderStatus"] {
            color: #64748b !important;
        }
        
        /* Smooth all transitions */
        * {
            transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* Disable problematic animations */
        .stFileUploader * {
            animation: none !important;
        }
        
        /* Force dark text on all uploader children - comprehensive override */
        .stFileUploader > *,
        .stFileUploader section,
        .stFileUploader section *,
        .stFileUploader [data-baseweb] * {
            color: #1e293b !important;
        }
        
        /* Text elements specifically - but exclude buttons */
        .stFileUploader p:not(button p),
        .stFileUploader span:not(button span),
        .stFileUploader div:not([class*="button"]):not([role="button"]):not(button) {
            color: #1e293b !important;
        }
        
        /* Ensure button text is always white */
        .stFileUploader button,
        .stFileUploader [role="button"][class*="button"],
        .stFileUploader [data-baseweb="button"] {
            background-color: #1e293b !important;
            color: white !important;
        }
        
        .stFileUploader button *,
        .stFileUploader [role="button"][class*="button"] *,
        .stFileUploader [data-baseweb="button"] * {
            color: white !important;
        }
        
        /* Exception: keep icons and buttons with appropriate colors */
        .stFileUploader svg {
            color: #64748b !important;
        }
        
        /* File size and name specifically */
        .stFileUploader [class*="file"],
        .stFileUploader [class*="File"] {
            color: #475569 !important;
        }
        
        /* CRITICAL: Override all previous rules for buttons - must be last */
        .stFileUploader button[type="button"],
        .stFileUploader [data-baseweb="button"],
        .stFileUploader [role="button"][data-baseweb] {
            background: #1e293b !important;
            background-color: #1e293b !important;
            color: white !important;
            border-color: #1e293b !important;
        }
        
        .stFileUploader button[type="button"] *,
        .stFileUploader [data-baseweb="button"] *,
        .stFileUploader [role="button"][data-baseweb] *,
        .stFileUploader button[type="button"] p,
        .stFileUploader button[type="button"] span,
        .stFileUploader button[type="button"] div {
            color: white !important;
            background: transparent !important;
        }
        
        .stFileUploader button[type="button"]:hover {
            background: #3b82f6 !important;
            background-color: #3b82f6 !important;
            border-color: #3b82f6 !important;
        }
        
        .stFileUploader button[type="button"]:hover * {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title='AI Radiology Report Generator | Medical Imaging Analysis',
        page_icon='🩺',
        layout='wide',
        initial_sidebar_state='expanded',
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': 'AI-powered radiology report generation using deep learning'
        }
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Hero Header
    st.markdown("""
        <div class="hero-header">
            <h1 class="hero-title">🩺 AI Radiology Report Generator</h1>
            <p class="hero-subtitle">Advanced Medical Imaging Analysis • CheXNet + LSTM Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        st.markdown("---")
        
        weights_h5 = st.text_input(
            'Decoder Weights',
            'encoder_decoder_epoch_5.weights.h5',
            help='Path to trained decoder model weights'
        )
        tokenizer_pkl = st.text_input(
            'Tokenizer Path',
            'models/tokenizer.pkl',
            help='Path to tokenizer pickle file'
        )
        chexnet_h5 = st.text_input(
            'CheXNet Weights',
            'brucechou1983_CheXNet_Keras_0.3.0_weights.h5',
            help='Path to CheXNet feature extractor weights'
        )
        
        st.markdown("---")
        st.markdown("### 🎛️ Generation Parameters")
        
        top_k = st.slider('Top-k Sampling', 1, 10, 5, 
                         help='Number of top predictions to sample from')
        temperature = st.slider('Temperature', 0.5, 1.5, 0.8, 0.05,
                               help='Controls randomness in generation')
        
        st.markdown("---")
        st.markdown("### 📚 Model Architecture")
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.1); 
                    padding: 1.25rem; 
                    border-radius: 12px; 
                    border-left: 4px solid rgba(255, 255, 255, 0.3);
                    backdrop-filter: blur(10px);'>
            <p style='color: white; font-weight: 600; margin: 0 0 0.75rem 0; font-size: 0.95rem;'>
                Deep Learning Pipeline
            </p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 0.85rem; line-height: 1.8;'>
                • DenseNet121 Feature Extraction<br>
                • LSTM Sequence Decoder<br>
                • GloVe Word Embeddings<br>
                • Trained on IU X-Ray Dataset<br>
                • Medical Report Generation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 1rem; color: rgba(255,255,255,0.7); font-size: 0.75rem;'>
            <p style='margin: 0;'>© 2024 AI Medical Imaging</p>
            <p style='margin: 0.25rem 0 0 0;'>Research & Education Only</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check model files
    required_artifacts = [weights_h5, tokenizer_pkl, chexnet_h5]
    if not all(path and os.path.exists(path) for path in required_artifacts):
        st.error('⚠️ Model files not found. Please ensure all required model files are available in the specified paths.')
        st.stop()
    
    # Patient Information
    st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input('Patient Full Name', 'John Doe', 
                                     help='Enter patient\'s complete name')
    with col2:
        patient_id = st.text_input('Patient ID', 'P-' + datetime.now().strftime('%Y%m%d-%H%M'),
                                   help='Unique patient identifier')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age (years)', '45', help='Patient age in years')
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'], 
                             help='Patient gender')
    with col3:
        exam_date = st.date_input('Examination Date', datetime.now(),
                                 help='Date of X-ray examination')
    
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    # Image Upload
    st.markdown('<div class="section-title">📸 Chest X-Ray Image Upload</div>', unsafe_allow_html=True)
    st.markdown("""
        <p style='color: #64748b !important; margin-bottom: 1.5rem; font-size: 0.95rem; line-height: 1.6;'>
            Upload frontal and/or lateral view chest X-ray images. The system accepts PNG, JPG, or JPEG formats.
            For optimal results, ensure images are clear and properly oriented.
        </p>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        'Select X-Ray Images',
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='upload_images',
        help='Upload 1-2 X-ray images (frontal/lateral views)'
    )
    
    uploaded_pair = []
    if uploaded_files:
        uploaded_pair = [{'name': f.name, 'data': f.getvalue()} for f in uploaded_files[:2]]
        if len(uploaded_pair) == 1:
            uploaded_pair *= 2
    
    if uploaded_pair:
        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div style='text-align: center; margin-bottom: 1rem;'>
                    <span style='font-weight: 600; color: #1e293b; font-size: 1.1rem;'>
                        📋 Frontal View
                    </span>
                </div>
            """, unsafe_allow_html=True)
            st.image(uploaded_pair[0]['data'], use_container_width=True)
        with col2:
            st.markdown("""
                <div style='text-align: center; margin-bottom: 1rem;'>
                    <span style='font-weight: 600; color: #1e293b; font-size: 1.1rem;'>
                        📋 Lateral View
                    </span>
                </div>
            """, unsafe_allow_html=True)
            st.image(uploaded_pair[1]['data'], use_container_width=True)
    
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner('🔄 Initializing AI models and neural networks...'):
        encoder_model, decoder_model, tokenizer = load_models(weights_h5, tokenizer_pkl)
        chexnet = load_chexnet(chexnet_h5)
    
    # Generate Button
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button('🚀 Generate Medical Report', use_container_width=True)
    
    if generate_button:
        if not uploaded_pair:
            st.error('❌ Please upload at least one X-ray image')
            st.stop()
        
        with st.spinner('🔬 Analyzing radiological images with deep learning...'):
            temp_paths = []
            try:
                for item in uploaded_pair[:2]:
                    suffix = Path(item['name']).suffix or '.png'
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(item['data'])
                    tmp.flush()
                    tmp.close()
                    temp_paths.append(tmp.name)
                
                feats = infer_features(chexnet, temp_paths[0], temp_paths[1])
                raw_report = generate_report(encoder_model, decoder_model, tokenizer, feats, 
                                            top_k=top_k, temperature=temperature)
                formatted_report = format_report_text(raw_report)
                
            finally:
                for path in temp_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
        
        st.success('✅ Medical report generated successfully!')
        
        # Display Report
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="report-title">📋 Medical Report</div>', unsafe_allow_html=True)
        
        # Patient Information Header
        st.markdown("""
            <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;
                        border-left: 4px solid #3b82f6;'>
                <h3 style='color: #1e40af; margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 600;'>
                    Patient Information
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Patient metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patient Name", patient_name)
        col2.metric("Patient ID", patient_id)
        col3.metric("Age", f"{age} years")
        col4.metric("Gender", gender)
        
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Exam Date", exam_date.strftime('%B %d, %Y'))
        col2.metric("Report Generated", datetime.now().strftime('%B %d, %Y'))
        col3.metric("Time", datetime.now().strftime('%I:%M %p'))
        
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        # Report content
        html_report = formatted_report.replace('**', '<strong>').replace('</strong><strong>', '').replace('\n', '<br>')
        st.markdown(f"""
            <div class="report-card">
                <div class="report-section">
                    <strong>EXAMINATION TYPE</strong>
                    <p style='margin: 0.5rem 0 0 0; color: #475569; line-height: 1.6;'>
                        Chest Radiography: Frontal and Lateral Projections
                    </p>
                </div>
                <div class="report-section">
                    {html_report}
                </div>
                <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 2px solid #e2e8f0;'>
                    <p style='color: #64748b; font-size: 0.85rem; margin: 0; text-align: center;'>
                        <strong>Disclaimer:</strong> This report was generated using AI-based analysis for research and educational purposes only. 
                        Clinical decisions should not be made based solely on this AI-generated report.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # PDF Download
        st.markdown('<div style="margin-top: 2.5rem;"></div>', unsafe_allow_html=True)
        
        if PDF_AVAILABLE:
            try:
                pdf_buffer = generate_pdf_report(
                    patient_name=patient_name,
                    patient_id=patient_id,
                    age=age,
                    gender=gender,
                    exam_date=exam_date.strftime('%Y-%m-%d'),
                    report_text=formatted_report,
                    logo_path='logo.png' if os.path.exists('logo.png') else None
                )
                
                st.markdown("""
                    <div style='text-align: center; margin-bottom: 1rem;'>
                        <p style='color: #475569; font-size: 0.95rem; margin: 0;'>
                            💾 Download a professionally formatted PDF version of this report
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label='📥 Download PDF Report',
                        data=pdf_buffer,
                        file_name=f'radiology_report_{patient_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                        mime='application/pdf',
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f'⚠️ PDF generation encountered an issue: {str(e)}')
        else:
            st.info('ℹ️ To enable PDF downloads, install reportlab: `pip install reportlab`')


if __name__ == '__main__':
    main()


