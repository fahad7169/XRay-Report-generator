import xml.etree.ElementTree as ET
import os
from collections import Counter

# Define keywords
normal_keywords = {'no', 'normal', 'clear', 'unremarkable', 'stable', 'within normal limits', 'free of'}
disease_keywords = {
    'pneumothorax', 'effusion', 'consolidation', 'atelectasis', 'edema', 
    'cardiomegaly', 'enlarged', 'opacity', 'infiltrate', 'cabg', 'sternotomy', 
    'abnormality', 'mass', 'nodule', 'granuloma', 'calcification', 'pleural',
    'blunting', 'consolidated', 'infiltrates', 'focal', 'thickening'
}

# Counters
normal_count = 0
disease_count = 0
keywords_found = Counter()
disease_found = Counter()
total_files = 0

# Get XML files
files = [f for f in os.listdir('ecgen-radiology') if f.endswith('.xml')]
total_files = len(files)
print(f'Total XML files: {total_files}')
print(f'Analyzing first 5000 files...\n')

# Analyze files
count = 0
for f in files[:5000]:
    try:
        count += 1
        path = f"ecgen-radiology/{f}"
        tree = ET.parse(path)
        
        # Extract text from FINDINGS and IMPRESSION
        findings = tree.find(".//AbstractText[@Label='FINDINGS']")
        impression = tree.find(".//AbstractText[@Label='IMPRESSION']")
        
        text = ""
        if findings is not None and findings.text:
            text += " " + findings.text.lower()
        if impression is not None and impression.text:
            text += " " + impression.text.lower()
        
        # Check for keywords
        has_normal = any(kw in text for kw in normal_keywords)
        has_disease = any(kw in text for kw in disease_keywords)
        
        if has_normal:
            normal_count += 1
        if has_disease:
            disease_count += 1
        
        # Count specific keywords
        for kw in normal_keywords:
            if kw in text:
                keywords_found[kw] += 1
        
        for kw in disease_keywords:
            if kw in text:
                disease_found[kw] += 1
                
    except Exception as e:
        if count % 1000 == 0:
            print(f"Processed {count} files...")

print(f'\nAnalysis Results (based on {count} reports):')
print('=' * 60)
print(f'Reports with "normal" keywords: {normal_count} ({100*normal_count/count:.1f}%)')
print(f'Reports with disease keywords: {disease_count} ({100*disease_count/count:.1f}%)')

print(f'\nDisease prevalence: {disease_count/(count-disease_count)*100:.1f}%')
print(f'Normal prevalence: {normal_count/(count-normal_count)*100:.1f}%')
print('\nTop 10 Normal Keywords:')
for word, freq in keywords_found.most_common(10):
    print(f'  {word}: {freq} times ({100*freq/count:.1f}% of reports)')

print('\nTop 15 Disease Keywords:')
for word, freq in disease_found.most_common(15):
    print(f'  {word}: {freq} times ({100*freq/count:.1f}% of reports)')

