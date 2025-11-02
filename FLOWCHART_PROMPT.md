# Flowchart Prompt for Gemini

## Ready-to-Use Prompt (Copy this to Gemini):

Create an original, professional flowchart for an automated chest X-ray radiology report generation system. Design it with a unique, modern medical/technical aesthetic that clearly visualizes the complete image-to-text pipeline. Be creative with the design while maintaining clarity and professionalism.

### Main Flow (Left to Right):

**Nodes:**
1. **"Chest X-ray Images"** (Input: Frontal + Lateral views)
   - Color: Light red/pink
   
2. **"Image Preprocessing"**
   - Color: Light yellow/orange
   - Shape: Hexagon (optional, to distinguish from other nodes)
   - Subtitle within box: "Resize 224×224, Normalize [0,1]"
   - Note: Use "×" (multiplication symbol) for dimensions
   
3. **"Feature Extraction"**
   - Color: Light purple (central node)
   - Subtitle within box: "CheXNet (DenseNet121): 1024-dim × 2 → Concatenate 2048-dim"
   - Note: Use "×" (multiplication symbol), not "x" or "*"
   
4. **"Encoder-Decoder Model"**
   - Color: Light green
   - Subtitle within box: "Encoder: 2048→256 | Decoder: LSTM×2 (256) + GloVe (300-dim)"
   - Note: Use "×" (multiplication symbol) and "→" (arrow) for dimensions
   
5. **"Generated Radiology Report"**
   - Shape: Document icon (paper with folded corner) or text document representation
   - Positioned to the right of last arrow
   - Text color: Black or dark blue

### Arrows with Labels:
- Chest X-ray Images → Image Preprocessing
  - Label on arrow: "Pixel Data"
- Image Preprocessing → Feature Extraction
  - Label on arrow: "Processed Images"
- Feature Extraction → Encoder-Decoder Model
  - Label on arrow: "2048-dim Features"
- Encoder-Decoder Model → Generated Radiology Report
  - Label on arrow: "Predicted Text Sequence"

### Small Auxiliary Boxes Connected to "Feature Extraction":

**Above Feature Extraction (pointing down with arrows):**
- **"CheXNet"** (Light blue rounded rectangle)
- **"DenseNet121"** (Light blue rounded rectangle)

**Below Feature Extraction (pointing up with arrows):**
- **"Frontal Image Features (1024-dim)"** (Light red/pink rounded rectangle)
- **"Lateral Image Features (1024-dim)"** (Light red/pink rounded rectangle)

### Additional Small Boxes Connected to "Encoder-Decoder Model":

**Above Encoder-Decoder Model (pointing down with arrows):**
- **"GloVe Embeddings (300-dim)"** (Light blue rounded rectangle)
- **"Tokenizer (OOV)"** (Light blue rounded rectangle)
  - Note: Use "OOV" not "BPE" - this is an Out-of-Vocabulary tokenizer

**Below Encoder-Decoder Model (pointing up with arrows):**
- **"Top-k Sampling"** (Light red/pink rounded rectangle)
- **"Temperature Scaling"** (Light red/pink rounded rectangle)

### Design Guidelines (Be Creative!):
- **Style**: Create your own unique, modern design aesthetic - don't copy any existing flowchart style
- **Layout**: Choose the best layout (horizontal, vertical, diagonal, or hybrid) that tells the story clearly
- **Node Shapes**: Use creative, meaningful shapes:
  - Input images: Could be image frames, X-ray silhouettes, or visual representations
  - Processing steps: Your choice - rectangles, hexagons, diamonds, circles, or custom shapes
  - Model components: Could be neural network visualizations, layered structures, or abstract designs
  - Output: Text/document representation that fits your unique design
- **Color Scheme**: Design a cohesive, professional palette:
  - Consider medical/healthcare aesthetics (but not limited to traditional colors)
  - Use color strategically to show relationships and progression
  - Ensure excellent readability and contrast
- **Connections**: 
  - Design visually interesting connectors (arrows, flow lines, curves, or unique patterns)
  - Show data transformations and dimensions clearly
  - Add helpful annotations on connections where needed
- **Typography**: 
  - Choose modern, readable fonts that match your design
  - Create clear hierarchy with sizing and weights
  - Include technical details appropriately
- **Visual Elements**: 
  - Add icons, symbols, or visual metaphors that enhance understanding
  - Use grouping and spacing creatively
  - Consider subtle backgrounds, shadows, or depth effects if it helps clarity
- **Innovation**: Feel free to add creative elements like:
  - Visual metaphors for medical imaging
  - Gradient effects or modern UI elements
  - Layered information presentation
  - Any design element that makes it more engaging while remaining professional

### Important Notes for Accuracy:
- **Tokenizer**: Use "Tokenizer (OOV)" - it uses Out-of-Vocabulary token, NOT BPE (Byte Pair Encoding)
- **Dimensions**: Use consistent notation - "1024-dim", "300-dim" (always use "-dim" suffix for consistency)
- **Symbols**: Use proper mathematical symbols: "×" (multiplication), "→" (arrow), "×2" (times 2)
- **Spelling**: Double-check all technical terms:
  - "CheXNet" (capital X)
  - "DenseNet121" (capital N, no spaces)
  - "LSTM×2" (multiplication symbol, not "x")
  - "Preprocessing" (not "Preporessing" or "Pre-processing")
- **Arrow Labels**: Include all arrow labels as specified above

### Output Requirements:
- High-resolution PNG format (at least 1920×1080 or higher)
- Clean, professional appearance suitable for:
  - Research papers
  - Presentations
  - Documentation
  - Conference posters
- All text clearly readable with correct spelling and technical accuracy
- Visually appealing and easy to understand at a glance

