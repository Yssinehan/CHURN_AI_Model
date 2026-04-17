import io
import datetime
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_NAMES
import joblib

def get_risk_level(prob):
    if prob < 0.3:
        return "Faible Risque", colors.green
    elif prob < 0.7:
        return "Risque Moyen", colors.orange
    else:
        return "Haut Risque", colors.red

def get_recommendations(pred, prob):
    if pred == 0:
        return [
            "• Maintenir la relation client habituelle",
            "• Surveillance standard des métriques d'usage",
            "• Campagnes promotionnelles saisonnières"
        ]
    elif prob < 0.5:
        return [
            "• Contact personnalisé dans les 7 jours",
            "• Offre fidélité (réduction 10-20%)",
            "• Suivi satisfaction après intervention"
        ]
    else:
        return [
            "• ACTION URGENTE: Appel dans 24h",
            "• Offre spéciale (réduction 30% + bonus)",
            "• Transfert équipe rétention prioritaire",
            "• Rendez-vous physique si possible"
        ]

def generate_pdf_report(client_data, prob, pred_class, feature_importances):
    """
    Generate 1-page professional PDF report.
    
    Args:
        client_data: dict with FEATURE_NAMES keys
        prob: churn probability [0,1]
        pred_class: 0 or 1
        feature_importances: pd.DataFrame('feature', 'importance')
    
    Returns:
        bytes: PDF content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Header
    today = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph(f"<b>Rapport d'Évaluation Risque Churn</b><br/><br/>Généré le: {today}", title_style))
    story.append(Spacer(1, 12))
    
    # Client Details Table
    story.append(Paragraph("1. Données Client", heading_style))
    client_table_data = [['Feature', 'Valeur']] + [[k, f"{v:.2f}" if isinstance(v, (int,float)) else str(v)] for k,v in client_data.items()]
    client_table = Table(client_table_data, colWidths=[6*cm, 8*cm])
    client_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(client_table)
    story.append(Spacer(1, 18))
    
    # Prediction Section
    risk_text, risk_color = get_risk_level(prob)
    pred_text = "Pas de Churn" if pred_class == 0 else "Churn Prédit"
    story.append(Paragraph("2. Résultats de Prédiction", heading_style))
    
    pred_para = f"<b>Probabilité Churn:</b> {prob:.1%}<br/>" \
                f"<b>Prédiction:</b> {pred_text}<br/>" \
                f"<b>Niveau de Risque:</b> {risk_text}"
    story.append(Paragraph(pred_para, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Top 5 Features
    top5 = feature_importances.nlargest(5, 'importance')
    fi_text = "<br/>".join([f"• {row['feature']}: {row['importance']:.1%}" for _, row in top5.iterrows()])
    story.append(Paragraph(f"3. Facteurs d'Influence Principaux (Top 5)<br/><br/>{fi_text}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Recommendations
    recs = get_recommendations(pred_class, prob)
    recs_text = "<br/>".join(recs)
    story.append(Paragraph("4. Recommandations d'Action<br/><br/>" + recs_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Footer
    footer = Paragraph(
        f"Modèle: Random Forest (Précision ~98%) | ChurnAi v1.0 | Confidentialiel",
        styles['Normal'], alignment=TA_CENTER
    )
    story.append(footer)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Test function (standalone)
if __name__ == "__main__":
    # Mock data
    mock_data = dict(zip(FEATURE_NAMES, [7,0,38,100,500,100,50,5,3,1,1,30,470]))
    importances = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': np.random.rand(len(FEATURE_NAMES))
    })
    pdf_bytes = generate_pdf_report(mock_data, 0.23, 0, importances)
    with open('test_report.pdf', 'wb') as f:
        f.write(pdf_bytes)
    print("Test PDF generated: test_report.pdf")
