"""
Generate real test PDF with table data
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.units import inch

def create_sample_table_pdf():
    """Create PDF with table containing amounts 300, 400, 550"""
    pdf_path = "test-data/sample_table.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    
    # Table data
    data = [
        ['Item', 'Amount'],
        ['Product A', '300'],
        ['Product B', '400'],
        ['Product C', '550']
    ]
    
    # Create table
    table = Table(data, colWidths=[2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    doc.build(elements)
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    create_sample_table_pdf()
