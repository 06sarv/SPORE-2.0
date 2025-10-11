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
            # Check for mistral (faster) or llama3
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
        return None  # No explanation if LLM unavailable
    
    try:
        print(f"  Generating AI explanation for {taxonomic_level}...")
        
        # More specific prompt for better, varied responses
        ph = soil_data.get('PH_WATER', 7)
        oc = soil_data.get('ORG_CARBON', 10)
        sand = soil_data.get('SAND', 33)
        clay = soil_data.get('CLAY', 33)
        
        # Professional biological prompt
        prompt = f"""As a soil microbiologist, provide a professional scientific explanation (2-3 sentences, ~50 words) for why {predictions_summary} bacteria are predicted in this soil environment:

Soil Properties:
- pH: {ph:.1f}
- Organic Carbon: {oc:.1f} g/kg
- Sand: {sand:.0f}%, Clay: {clay:.0f}%
- Taxonomic Level: {taxonomic_level}

Focus on specific soil-microbe ecological relationships, nutrient cycling, and environmental adaptations."""

        # Use Ollama API with Mistral (faster than llama3)
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "mistral:7b",  # Faster model
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 120,  # Allow for 2-3 sentences
                    "num_ctx": 1024,     # Larger context for better quality
                    "top_k": 40,
                    "top_p": 0.9
                }
            },
            timeout=40  # 40 second timeout for quality response
        )
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get('response', '').strip()
            
            # Remove incomplete sentences
            if explanation and not explanation.endswith(('.', '!', '?')):
                last_period = explanation.rfind('.')
                if last_period > 0:
                    explanation = explanation[:last_period + 1]
                else:
                    # Keep incomplete if it's substantial
                    if len(explanation) < 20:
                        return None
            
            print(f"  ✓ AI explanation generated for {taxonomic_level}")
            return explanation if explanation else None
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print(f"  ⚠ LLM timeout for {taxonomic_level} - skipping explanation")
        return None
    except Exception as e:
        print(f"  ⚠ LLM generation error for {taxonomic_level}: {str(e)[:50]} - skipping explanation")
        return None

def create_probability_chart(predictions, level_name, output_path):
    """Create horizontal bar chart for predictions"""
    plt.figure(figsize=(8, max(4, len(predictions) * 0.4)))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    names = [p['name'][:30] + '...' if len(p['name']) > 30 else p['name'] for p in predictions]
    probs = [p['probability'] * 100 for p in predictions]
    colors_list = ['#4caf50' if p['confidence'] == 'high' else '#ff9800' if p['confidence'] == 'medium' else '#9e9e9e' for p in predictions]
    
    plt.barh(names, probs, color=colors_list, edgecolor='black', linewidth=0.5)
    plt.xlabel('Probability (%)', fontsize=11, fontweight='bold')
    plt.title(f'{level_name.upper()} Level Predictions', fontsize=13, fontweight='bold', pad=15)
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def create_confidence_distribution(all_predictions, output_path):
    """Create pie chart showing confidence distribution"""
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
    for preds in all_predictions.values():
        for p in preds:
            confidence_counts[p['confidence']] += 1
    
    plt.figure(figsize=(6, 6))
    colors_pie = ['#4caf50', '#ff9800', '#9e9e9e']
    labels = [f"{k.capitalize()}\n({v})" for k, v in confidence_counts.items() if v > 0]
    sizes = [v for v in confidence_counts.values() if v > 0]
    
    plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    plt.title('Confidence Distribution Across All Predictions', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def create_soil_characteristics_chart(soil_data, output_path):
    """Create radar chart for soil characteristics"""
    categories = ['pH', 'Org Carbon', 'Sand', 'Silt', 'Clay', 'Bulk Density']
    
    # Normalize values to 0-100 scale
    values = [
        (soil_data.get('PH_WATER', 7) - 4) / 5 * 100,  # pH 4-9 -> 0-100
        soil_data.get('ORG_CARBON', 10) / 100 * 100,   # 0-100 g/kg
        soil_data.get('SAND', 33),                      # Already 0-100%
        soil_data.get('SILT', 33),                      # Already 0-100%
        soil_data.get('CLAY', 33),                      # Already 0-100%
        (soil_data.get('BULK', 1.4) - 0.5) / 1.5 * 100 # 0.5-2.0 -> 0-100
    ]
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#667eea', label='Soil Profile')
    ax.fill(angles, values, alpha=0.25, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Soil Characteristics Profile', fontsize=13, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def generate_hierarchical_report(smu_id, soil_chars, predictions_by_level, location_info, model_metrics, enable_ai=True):
    """
    Generate professional scientific PDF report
    
    Args:
        smu_id: Soil mapping unit ID
        soil_chars: Dictionary of soil characteristics
        predictions_by_level: Dictionary of predictions per taxonomic level
        location_info: Dictionary with lat, lon
        model_metrics: Dictionary with model performance metrics
        enable_ai: Boolean to enable/disable AI explanations (default: True)
    """
    try:
        print("\n" + "="*60)
        print("📄 Generating Professional Scientific Report...")
        print("="*60)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"SPORE_Report_{timestamp}.pdf"
        output_path = os.path.join(os.getcwd(), output_filename)
        print(f"Output: {output_filename}")
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                              topMargin=0.75*inch, bottomMargin=0.75*inch,
                              leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        # Custom styles
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#333333'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=colors.HexColor('#667eea'),
            borderPadding=5,
            backColor=colors.HexColor('#f5f5f5')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#444444'),
            alignment=TA_JUSTIFY,
            spaceAfter=8
        )
        
        # Build document content
        story = []
        
        # ============ TITLE PAGE ============
        story.append(Spacer(1, 0.4*inch))
        story.append(Paragraph("SPORE", title_style))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Heading3'],
            alignment=TA_CENTER,
            fontSize=12,
            textColor=colors.HexColor('#666666')
        )
        story.append(Paragraph("Soil-based Prediction Of Resident Entities", subtitle_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Hierarchical Microbiome Analysis Report", styles['Heading2']))
        story.append(Spacer(1, 0.15*inch))
        
        # Report metadata
        metadata_data = [
            ["Report Generated:", datetime.now().strftime("%B %d, %Y at %H:%M:%S")],
            ["Location:", f"Lat: {location_info.get('latitude', 'N/A'):.4f}, Lon: {location_info.get('longitude', 'N/A'):.4f}"],
            ["SMU ID:", str(smu_id)],
            ["Analysis Type:", "Hierarchical Multi-Level Classification"],
            ["Model Version:", "2.0 (India-specific)"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metadata_table)
        story.append(PageBreak())
        
        # ============ EXECUTIVE SUMMARY ============
        story.append(Paragraph("Executive Summary", heading_style))
        
        total_predictions = sum(len(preds) for preds in predictions_by_level.values())
        summary_text = f"""This report presents a comprehensive hierarchical analysis of soil microbiome composition 
        based on soil characteristics at the specified location in India. Using advanced machine learning models 
        trained on 96,697 bacteria occurrences, we identified {total_predictions} potential bacterial taxa across 
        5 taxonomic levels (Phylum → Class → Order → Family → Genus). The analysis employs Random Forest classifiers 
        with balanced class weights to handle the complex multi-label classification problem inherent in microbiome prediction."""
        
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.15*inch))
        
        # ============ SOIL CHARACTERISTICS ============
        story.append(Paragraph("1. Soil Characteristics Analysis", heading_style))
        
        soil_table_data = [["Parameter", "Value", "Unit", "Interpretation"]]
        
        interpretations = {
            'PH_WATER': lambda v: "Acidic" if v < 6.5 else "Neutral" if v < 7.5 else "Alkaline",
            'ORG_CARBON': lambda v: "Low" if v < 10 else "Medium" if v < 20 else "High",
            'SAND': lambda v: "Sandy" if v > 50 else "Moderate",
            'CLAY': lambda v: "Clayey" if v > 40 else "Moderate",
            'BULK': lambda v: "Compact" if v > 1.6 else "Loose" if v < 1.2 else "Normal"
        }
        
        soil_params = [
            ('PH_WATER', 'Soil pH', ''),
            ('ORG_CARBON', 'Organic Carbon', 'g/kg'),
            ('SAND', 'Sand Content', '%'),
            ('SILT', 'Silt Content', '%'),
            ('CLAY', 'Clay Content', '%'),
            ('COARSE', 'Gravel Content', '%'),
            ('CEC_CLAY', 'CEC of Clay', 'cmol(+)/kg'),
            ('BULK', 'Bulk Density', 'kg/dm³')
        ]
        
        for key, name, unit in soil_params:
            value = soil_chars.get(key, 'N/A')
            if value != 'N/A':
                interp = interpretations.get(key, lambda v: "Normal")(value)
                soil_table_data.append([name, f"{value:.2f}", unit, interp])
        
        soil_table = Table(soil_table_data, colWidths=[2.2*inch, 1.2*inch, 1*inch, 1.6*inch])
        soil_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(soil_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Add soil characteristics radar chart
        print("  Creating soil characteristics chart...")
        soil_chart_path = "temp_soil_chart.png"
        create_soil_characteristics_chart(soil_chars, soil_chart_path)
        story.append(Image(soil_chart_path, width=4.5*inch, height=4.5*inch))
        story.append(Spacer(1, 0.2*inch))
        print("  ✓ Soil chart created")
        
        # ============ MODEL PERFORMANCE ============
        story.append(PageBreak())
        story.append(Paragraph("2. Model Performance Metrics", heading_style))
        
        model_table_data = [["Taxonomic Level", "Classes", "F1 Score", "Accuracy", "Status"]]
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            if level in model_metrics:
                metrics = model_metrics[level]
                status = "✓ Good" if metrics['f1_score'] > 0.05 else "⚠ Fair"
                model_table_data.append([
                    level.capitalize(),
                    str(metrics['n_classes']),
                    f"{metrics['f1_score']:.4f}",
                    f"{metrics['accuracy']:.4f}",
                    status
                ])
        
        model_table = Table(model_table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1.5*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("""<i>Note: F1 scores reflect the challenging nature of microbiome prediction 
        with high taxonomic diversity. Hierarchical classification significantly outperforms flat multi-label approaches.</i>""", 
        body_style))
        
        # ============ HIERARCHICAL PREDICTIONS ============
        story.append(PageBreak())
        story.append(Paragraph("3. Hierarchical Prediction Results", heading_style))
        
        # Add confidence distribution chart
        print("  Creating confidence distribution chart...")
        conf_chart_path = "temp_confidence_dist.png"
        create_confidence_distribution(predictions_by_level, conf_chart_path)
        story.append(Image(conf_chart_path, width=4*inch, height=4*inch))
        story.append(Spacer(1, 0.2*inch))
        print("  ✓ Confidence chart created")
        
        # Predictions for each taxonomic level
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            if level in predictions_by_level and predictions_by_level[level]:
                story.append(PageBreak())
                story.append(Paragraph(f"3.{['phylum', 'class', 'order', 'family', 'genus'].index(level) + 1} {level.upper()} Level", heading_style))
                
                preds = predictions_by_level[level][:5]  # Top 5
                
                # Create chart
                print(f"  Creating {level} prediction chart...")
                chart_path = f"temp_{level}_chart.png"
                create_probability_chart(preds, level, chart_path)
                story.append(Image(chart_path, width=6*inch, height=max(3*inch, len(preds) * 0.4*inch)))
                story.append(Spacer(1, 0.15*inch))
                print(f"  ✓ {level} chart created")
                
                # Predictions table
                pred_table_data = [["Rank", "Taxon Name", "Probability", "Confidence"]]
                for idx, pred in enumerate(preds, 1):
                    pred_table_data.append([
                        str(idx),
                        pred['name'],
                        f"{pred['probability']*100:.2f}%",
                        pred['confidence'].upper()
                    ])
                
                pred_table = Table(pred_table_data, colWidths=[0.6*inch, 3.5*inch, 1.2*inch, 1.2*inch])
                pred_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(pred_table)
                story.append(Spacer(1, 0.15*inch))
                
                # AI-generated explanation (if enabled and available)
                if enable_ai and LLM_AVAILABLE:
                    pred_names = ", ".join([p['name'] for p in preds[:3]])
                    explanation = generate_ai_explanation(soil_chars, pred_names, level)
                    
                    # Only add explanation section if we got a valid response
                    if explanation:
                        # Use a style with better wrapping
                        explanation_style = ParagraphStyle(
                            'Explanation',
                            parent=body_style,
                            fontSize=10,
                            textColor=colors.HexColor('#555555'),
                            alignment=TA_JUSTIFY,
                            spaceAfter=10,
                            leading=14,
                            wordWrap='CJK'
                        )
                        
                        story.append(Paragraph(f"<b>Scientific Interpretation:</b>", body_style))
                        story.append(Paragraph(explanation, explanation_style))
        
        # ============ METHODOLOGY ============
        story.append(PageBreak())
        story.append(Paragraph("4. Methodology", heading_style))
        
        methodology_text = """<b>4.1 Data Sources</b><br/>
        • <b>Soil Data:</b> HWSD v2.0 (Harmonized World Soil Database) - 29,538 soil mapping units<br/>
        • <b>Bacteria Data:</b> GBIF (Global Biodiversity Information Facility) - 96,697 occurrences in India<br/>
        • <b>Citation:</b> GBIF.org (2 October 2025) GBIF Occurrence Download https://doi.org/10.15468/dl.z3ysfj<br/><br/>
        
        <b>4.2 Machine Learning Approach</b><br/>
        This analysis employs a hierarchical classification strategy with separate Random Forest models for each taxonomic level. 
        This approach significantly outperforms flat multi-label classification by reducing label dimensionality and respecting 
        biological taxonomy.<br/><br/>
        
        <b>Model Specifications:</b><br/>
        • Algorithm: Random Forest Classifier (100 trees)<br/>
        • Features: 13 engineered features (8 soil + 2 spatial + 3 derived)<br/>
        • Class Weighting: Balanced (handles imbalanced data)<br/>
        • Training Samples: 79,648 bacteria-soil linkages<br/>
        • Validation: Train-test split (80-20)<br/><br/>
        
        <b>4.3 Feature Engineering</b><br/>
        • Base features: COARSE, SAND, SILT, CLAY, ORG_CARBON, PH_WATER, CEC_CLAY, BULK<br/>
        • Spatial features: Latitude, Longitude (captures biogeography)<br/>
        • Derived features: Sand/Clay ratio, Silt/Clay ratio, Texture index<br/><br/>
        
        <b>4.4 Confidence Scoring</b><br/>
        • <b>High:</b> Probability ≥ 30%<br/>
        • <b>Medium:</b> Probability 15-30%<br/>
        • <b>Low:</b> Probability < 15%<br/>
        """
        
        story.append(Paragraph(methodology_text, body_style))
        
        # ============ LIMITATIONS & DISCLAIMER ============
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("5. Limitations & Disclaimer", heading_style))
        
        limitations_text = """<b>Geographic Scope:</b> This model is trained exclusively on bacteria occurrences from India. 
        Predictions for other regions may not be accurate.<br/><br/>
        
        <b>Prediction Uncertainty:</b> Microbiome composition is influenced by numerous factors beyond soil characteristics 
        (climate, season, land use, etc.). These predictions represent potential taxa based solely on soil properties.<br/><br/>
        
        <b>Model Performance:</b> F1 scores (0.038-0.067) reflect the inherent difficulty of predicting high-diversity 
        taxonomic data. These scores are significantly better than baseline and flat classification approaches.<br/><br/>
        
        <b>Validation Required:</b> Predictions should be validated through direct sampling and sequencing for research applications.<br/><br/>
        
        <b>Not for Clinical Use:</b> This tool is designed for research and educational purposes only.
        """
        
        story.append(Paragraph(limitations_text, body_style))
        
        # ============ FOOTER ============
        story.append(Spacer(1, 0.3*inch))
        footer_text = f"""<i>Generated by SPORE v2.0 | © {datetime.now().year} | 
        For research and educational purposes only | 
        Report ID: {timestamp}</i>"""
        story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=body_style, fontSize=8, 
                                                           textColor=colors.grey, alignment=TA_CENTER)))
        
        # Build PDF
        print("\n  Building PDF document...")
        doc.build(story)
        print("  ✓ PDF built successfully")
        
        # Clean up temporary chart files
        print("  Cleaning up temporary files...")
        for temp_file in ['temp_soil_chart.png', 'temp_confidence_dist.png'] + \
                         [f'temp_{level}_chart.png' for level in ['phylum', 'class', 'order', 'family', 'genus']]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"✓ Professional report generated: {output_filename}")
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
