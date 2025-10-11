from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

def generate_microbe_report(output_path, microbe_data, predictions, explanation):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Microbe Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Input Data
    story.append(Paragraph("<b>Input Microbe Data:</b>", styles['h2']))
    for key, value in microbe_data.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Predictions
    story.append(Paragraph("<b>Predictions:</b>", styles['h2']))
    if predictions:
        prediction_data = [["Microbe Name", "Probability", "Explanation"]]
        for microbe in predictions:
            prediction_data.append([
                microbe.get('name', 'N/A'),
                f"{microbe.get('probability', 0.0) * 100:.2f}%",
                microbe.get('explanation', 'N/A')
            ])
        
        table = Table(prediction_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No predictions available.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Explanation
    story.append(Paragraph("<b>Explanation:</b>", styles['h2']))
    story.append(Paragraph(explanation, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    doc.build(story)

if __name__ == '__main__':
    # Example Usage
    sample_microbe_data = {
        "Microbe Name": "Escherichia coli",
        "Habitat": "Intestines",
        "Type": "Bacteria"
    }
    sample_predictions = {
        "Pathogenicity": "Opportunistic",
        "Growth Rate": "Fast"
    }
    sample_explanation = "Escherichia coli is a gram-negative, rod-shaped bacterium that is commonly found in the lower intestine of warm-blooded organisms. Most E. coli strains are harmless, but some can cause serious food poisoning in humans, and are occasionally responsible for product recalls due to food contamination."

    generate_microbe_report("sample_report.pdf", sample_microbe_data, sample_predictions, sample_explanation)
    print("Sample report generated at sample_report.pdf")