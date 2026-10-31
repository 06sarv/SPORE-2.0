"""
Professional scientific report generator for SPORE.
Builds a concise PDF summary with charts, model metadata, and optional
offline AI interpretation through Ollama when available.
"""

import os
from pathlib import Path
from datetime import datetime

os.environ.setdefault('MPLCONFIGDIR', '/tmp/spore-mplconfig')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/spore-cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import requests

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
        
        prompt = f"""Write a brief, cautious, model-generated hypothesis about these ranked taxa: {predictions_summary}.

Soil properties: pH {ph:.1f}; organic carbon {oc:.1f} g/kg; sand {sand:.0f}%; clay {clay:.0f}%. Taxonomic level: {taxonomic_level}.

Use 2 short sentences. Explain that the list is a model ranking based on these inputs. Do not claim the taxa were detected or are present, and do not assert taxon-specific ecology, prevalence, or causal mechanisms without supporting reference data. State that lab sampling is needed to verify presence."""

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
            timeout=120
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

def _format_coord(value, default='N/A'):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return default


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
        else:
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def generate_hierarchical_report(
    smu_id,
    soil_chars,
    predictions_by_level,
    location_info,
    model_metrics,
    model_info=None,
    enable_ai=True,
):
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
                              topMargin=0.4*inch, bottomMargin=0.35*inch,
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
        story.append(Paragraph(
            f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} | "
            f"Location: {_format_coord(location_info.get('latitude'))}°, {_format_coord(location_info.get('longitude'))}° | "
            f"SMU ID: {smu_id}",
            subtitle_style
        ))
        story.append(Spacer(1, 0.1*inch))

        model_name = (model_info or {}).get('model_type', 'Hierarchical model')
        model_version = (model_info or {}).get('version', 'Unknown')
        training_samples = (model_info or {}).get('training_samples', 'Unknown')
        feature_count = len((model_info or {}).get('feature_names', [])) or len(soil_chars)

        summary_table = Table([
            ['Model', model_name],
            ['Version', str(model_version)],
            ['Features', str(feature_count)],
            ['Training samples', str(training_samples)],
            ['H3 resolution', str(location_info.get('h3_resolution', 'N/A'))],
            ['H3 cell', str(location_info.get('h3_cell', 'N/A'))[:32]],
        ], colWidths=[1.35*inch, 4.85*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#111827')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#d1d5db')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.08*inch))
        
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
        story.append(Image(chart_path, width=6.5*inch, height=1.75*inch))
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
                pred_names = ", ".join([p['name'] for p in predictions_by_level['genus'][:2]])
                explanation = generate_ai_explanation(soil_chars, pred_names, 'genus')
                
                if explanation:
                    story.append(Paragraph("AI-Generated Hypothesis (not a detection)", heading_style))
                    explanation_style = ParagraphStyle('Explanation', parent=body_style, 
                                                      fontSize=8, alignment=TA_JUSTIFY, leading=10)
                    story.append(Paragraph(explanation, explanation_style))
                    story.append(Spacer(1, 0.08*inch))
        
        # ============ MODEL INFO & METHODOLOGY (ULTRA COMPACT) ============
        story.append(Paragraph("Model Info & Methodology", heading_style))
        
        phylum_f1 = model_metrics.get('phylum', {}).get('weighted_f1', model_metrics.get('phylum', {}).get('f1_score', 0))
        class_f1 = model_metrics.get('class', {}).get('weighted_f1', model_metrics.get('class', {}).get('f1_score', 0))
        genus_f1 = model_metrics.get('genus', {}).get('weighted_f1', model_metrics.get('genus', {}).get('f1_score', 0))
        source_names = ', '.join((model_info or {}).get('source_records', {}).keys()) or (model_info or {}).get('data_source', 'soil and occurrence training records')
        validation_note = model_info.get('spatial_validation', 'See model metadata for validation protocol.')
        methodology_text = f"""<b>Model:</b> {model_name} ({model_version}). <b>Data:</b> {source_names}.
        <b>Features:</b> {feature_count} engineered environmental variables.
        <b>Performance:</b> Phylum F1={phylum_f1:.3f}, Class F1={class_f1:.3f}, Genus F1={genus_f1:.3f}.
        <b>Validation:</b> {validation_note}
        <b>Disclaimer:</b> Predictions are probabilistic and reflect the model training distribution, not ground truth.
        Microbiome composition also depends on climate, season, land use, and sampling bias."""
        
        story.append(Paragraph(methodology_text, body_style))
        
        # ============ FOOTER ============
        story.append(Spacer(1, 0.03*inch))
        footer_text = f"""<i>SPORE v{model_version} | © {datetime.now().year} | For research use only | Report ID: {timestamp}</i>"""
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
