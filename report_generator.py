"""
Professional Scientific Report Generator for SPORE
Generates comprehensive PDF reports with visualizations and AI-powered explanations
Uses offline GPT4All for intelligent insights
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from datetime import datetime
import os
import numpy as np
import requests
import json

# Initialize Ollama LLM (offline, much faster than GPT4All)
# Uses Ollama API with llama3 - keeps model loaded in memory
ENABLE_LLM = True
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def check_ollama():
    """Check if Ollama is available and running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any('mistral' in model.get('name', '') or 'llama3' in model.get('name', '') for model in models)
        return False
    except:
        return False

if ENABLE_LLM:
    LLM_AVAILABLE = check_ollama()
    if LLM_AVAILABLE:
        print("✓ Ollama API available for report generation (mistral:7b - optimized for speed)")
    else:
        print("⚠ Ollama not available. Start with: ollama serve")
        LLM_AVAILABLE = False
else:
    LLM_AVAILABLE = False
    print("ℹ LLM disabled (set ENABLE_LLM=True to enable)")

def generate_ai_explanation(soil_data, predictions_summary, taxonomic_level):
    """Generate intelligent explanation using offline LLM - fully dynamic based on input/output"""
    if not LLM_AVAILABLE:
        return None
    
    try:
        ph = soil_data.get('PH_WATER', 7)
        oc = soil_data.get('ORG_CARBON', 10)
        sand = soil_data.get('SAND', 33)
        clay = soil_data.get('CLAY', 33)
        
        prompt = f"""As a soil microbiologist, provide a professional scientific explanation (2-3 sentences, ~50 words) for why {predictions_summary} bacteria are predicted in this soil environment:

Soil Properties:
- pH: {ph:.1f}
- Organic Carbon: {oc:.1f} g/kg
- Sand: {sand:.0f}%, Clay: {clay:.0f}%
- Taxonomic Level: {taxonomic_level}

Focus on specific soil-microbe ecological relationships, nutrient cycling, and environmental adaptations."""

        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 120,
                    "num_ctx": 1024,
                    "top_k": 40,
                    "top_p": 0.9
                }
            },
            timeout=40
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get('response', '').strip()
            
            if explanation and not explanation.endswith(('.', '!', '?')):
                last_period = explanation.rfind('.')
                if last_period > 0:
                    explanation = explanation[:last_period + 1]
                else:
                    if len(explanation) < 20:
                        return None
            
            return explanation if explanation else None
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None

def create_compact_predictions_chart(predictions_by_level, output_path):
    """Create compact multi-level predictions chart"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    levels_to_show = ['phylum', 'order', 'genus']
    
    for idx, level in enumerate(levels_to_show):
        if level in predictions_by_level:
            preds = predictions_by_level[level][:3]  # Top 3 only
            names = [p['name'][:20] + '...' if len(p['name']) > 20 else p['name'] for p in preds]
            probs = [p['probability'] * 100 for p in preds]
            colors_list = ['#4caf50' if p['confidence'] == 'high' else '#ff9800' for p in preds]
            
            axes[idx].barh(names, probs, color=colors_list, edgecolor='black', linewidth=0.5)
            axes[idx].set_xlabel('Prob (%)', fontsize=8)
            axes[idx].set_title(level.capitalize(), fontsize=9, fontweight='bold')
            axes[idx].set_xlim(0, 100)
            axes[idx].tick_params(axis='both', labelsize=7)
            axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def generate_hierarchical_report(smu_id, soil_chars, predictions_by_level, location_info, model_metrics, enable_ai=True):
    """Generate concise one-page professional scientific PDF report"""
    try:
        print("\n" + "="*60)
        print("📄 Generating One-Page Report...")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"SPORE_Report_{timestamp}.pdf"
        output_path = os.path.join(os.getcwd(), output_filename)
        print(f"Output: {output_filename}")
        
        # Tighter margins for one page
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.6*inch, rightMargin=0.6*inch)
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=4,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=4,
            spaceBefore=6,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#f5f5f5')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#444444'),
            alignment=TA_JUSTIFY,
            spaceAfter=4,
            leading=10
        )
        
        story = []
        
        # ============ COMPACT HEADER ============
        story.append(Paragraph("SPORE - Soil Microbiome Analysis", title_style))
        
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], alignment=TA_CENTER, 
                                       fontSize=9, textColor=colors.HexColor('#666666'))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} | "
                             f"Location: {location_info.get('latitude', 'N/A'):.4f}°N, {location_info.get('longitude', 'N/A'):.4f}°E | "
                             f"SMU ID: {smu_id}", subtitle_style))
        story.append(Spacer(1, 0.1*inch))
        
        # ============ SOIL CHARACTERISTICS (COMPACT TABLE) ============
        story.append(Paragraph("Soil Characteristics", heading_style))
        
        soil_table_data = [["Parameter", "Value", "Parameter", "Value"]]
        soil_params = [
            ('PH_WATER', 'pH', ''),
            ('ORG_CARBON', 'Org C', 'g/kg'),
            ('SAND', 'Sand', '%'),
            ('CLAY', 'Clay', '%'),
        ]
        
        # Create 2-column layout
        for i in range(0, len(soil_params), 2):
            row = []
            for j in range(2):
                if i + j < len(soil_params):
                    key, name, unit = soil_params[i + j]
                    value = soil_chars.get(key, 'N/A')
                    row.extend([name, f"{value:.1f} {unit}" if value != 'N/A' else 'N/A'])
                else:
                    row.extend(['', ''])
            soil_table_data.append(row)
        
        soil_table = Table(soil_table_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        soil_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(soil_table)
        story.append(Spacer(1, 0.08*inch))
        
        # ============ PREDICTIONS (COMPACT CHART) ============
        story.append(Paragraph("Top Predictions by Taxonomic Level", heading_style))
        
        print("  Creating compact predictions chart...")
        chart_path = "temp_compact_predictions.png"
        create_compact_predictions_chart(predictions_by_level, chart_path)
        story.append(Image(chart_path, width=6.5*inch, height=2*inch))
        story.append(Spacer(1, 0.08*inch))
        print("  ✓ Chart created")
        
        # ============ TOP PREDICTIONS TABLE ============
        story.append(Paragraph("Detailed Top Predictions", heading_style))
        
        pred_table_data = [["Level", "Taxon", "Probability", "Conf"]]
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            if level in predictions_by_level and predictions_by_level[level]:
                pred = predictions_by_level[level][0]  # Top 1 per level
                pred_table_data.append([
                    level.capitalize()[:3],
                    pred['name'][:35] + '...' if len(pred['name']) > 35 else pred['name'],
                    f"{pred['probability']*100:.1f}%",
                    pred['confidence'][0].upper()
                ])
        
        pred_table = Table(pred_table_data, colWidths=[0.6*inch, 3.8*inch, 1*inch, 0.6*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.08*inch))
        
        # ============ AI INTERPRETATION (COMPACT) ============
        if enable_ai and LLM_AVAILABLE:
            # Get interpretation for genus level (most specific)
            if 'genus' in predictions_by_level and predictions_by_level['genus']:
                story.append(Paragraph("Scientific Interpretation", heading_style))
                pred_names = ", ".join([p['name'] for p in predictions_by_level['genus'][:2]])
                explanation = generate_ai_explanation(soil_chars, pred_names, 'genus')
                
                if explanation:
                    explanation_style = ParagraphStyle('Explanation', parent=body_style, 
                                                      fontSize=8, alignment=TA_JUSTIFY, leading=10)
                    story.append(Paragraph(explanation, explanation_style))
                    story.append(Spacer(1, 0.08*inch))
        
        # ============ MODEL INFO & METHODOLOGY (ULTRA COMPACT) ============
        story.append(Paragraph("Model Info & Methodology", heading_style))
        
        methodology_text = f"""<b>Model:</b> Random Forest (hierarchical, 5 levels). <b>Data:</b> HWSD v2.0 soil DB (29,538 SMUs) + 
        GBIF bacteria (96,697 occurrences, India). <b>Features:</b> 13 engineered (8 soil + 2 spatial + 3 derived). 
        <b>Performance:</b> Phylum F1={model_metrics.get('phylum', {}).get('f1_score', 0):.3f}, 
        Genus F1={model_metrics.get('genus', {}).get('f1_score', 0):.3f}. <b>Citation:</b> GBIF.org (Oct 2025) 
        https://doi.org/10.15468/dl.z3ysfj. <b>Disclaimer:</b> Predictions based on soil properties only. 
        Microbiome influenced by climate, season, land use. For research only."""
        
        story.append(Paragraph(methodology_text, body_style))
        
        # ============ FOOTER ============
        story.append(Spacer(1, 0.1*inch))
        footer_text = f"""<i>SPORE v2.0 | © {datetime.now().year} | For research use only | Report ID: {timestamp}</i>"""
        story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=body_style, fontSize=7, 
                                                           textColor=colors.grey, alignment=TA_CENTER)))
        
        # Build PDF
        print("\n  Building one-page PDF...")
        doc.build(story)
        print("  ✓ PDF built successfully")
        
        # Cleanup
        if os.path.exists(chart_path):
            os.remove(chart_path)
        
        print(f"✓ One-page report generated: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"✗ Report generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Legacy function for backward compatibility
def generate_microbe_report(output_path, microbe_data, predictions, explanation):
    """Legacy function - redirects to new hierarchical report"""
    print("⚠ Using legacy report function - consider updating to generate_hierarchical_report()")
    return output_path