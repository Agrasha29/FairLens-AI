from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(file_path, fairness, fairness_score, gemini_text):

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("FairLens AI - Bias Audit Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Fairness Score: {fairness_score}/100", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Male Selection Rate: {fairness['Male Selection Rate']}", styles["Normal"]))
    content.append(Paragraph(f"Female Selection Rate: {fairness['Female Selection Rate']}", styles["Normal"]))
    content.append(Paragraph(f"Disparate Impact: {fairness['Disparate Impact Ratio']}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("AI Explanation:", styles["Heading2"]))
    content.append(Paragraph(str(gemini_text), styles["Normal"]))

    doc.build(content)