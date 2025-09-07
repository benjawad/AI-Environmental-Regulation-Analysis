# pdf_generator.py  (replace your current file with this version)
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, PageBreak, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import landscape, A3, A4
from reportlab.lib.units import inch, mm, cm            # <-- Added cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from PyPDF2 import PdfMerger
import os
import tempfile
import shutil

script_dir = Path(__file__).resolve()
base_dir = script_dir.parent.parent.parent.parent
logo_path = base_dir / "media" / "jesa_logo.png"


def header_footer(canvas, doc):
    """Draw header / footer on each page."""
    jesa_blue = colors.Color(red=0/255, green=51/255, blue=102/255)

    canvas.saveState()
    page_w, page_h = landscape(A3)

    # top separator
    line_y = page_h - doc.topMargin + 0.5 * cm
    canvas.setStrokeColor(jesa_blue)
    canvas.setLineWidth(1)
    canvas.line(doc.leftMargin, line_y, page_w - doc.rightMargin, line_y)

    # logo area (if present)
    logo_w = 4.0 * cm
    logo_h = 3.0 * cm
    logo_y = line_y + 0.2 * cm
    try:
        if os.path.exists(logo_path):
            logo = Image(str(logo_path), width=logo_w, height=logo_h)
            logo.drawOn(canvas, doc.leftMargin, logo_y)
        else:
            # fallback text if logo not present
            canvas.setFont('Helvetica-Bold', 18)
            canvas.setFillColor(jesa_blue)
            canvas.drawString(doc.leftMargin, logo_y + 0.3 * cm, "JESA")
    except Exception:
        # Don't crash the whole PDF for a logo loading error
        canvas.setFont('Helvetica-Bold', 18)
        canvas.setFillColor(jesa_blue)
        canvas.drawString(doc.leftMargin, logo_y + 0.3 * cm, "JESA")

    # Left info table
    info_y = line_y - 0.2 * cm
    left_data = [
        ['Project Name:', 'Chemical additives plant'],
        ['Customer:', 'NOVADDIX'],
        ['Document Title:', 'Sustainable Project Delivery - Legal Register - Chemical additives plant']
    ]
    left_col_w = [(doc.width - logo_w - 0.5 * cm) * 0.1, (doc.width - logo_w - 0.5 * cm) * 0.8]
    left_tbl = Table(left_data, colWidths=left_col_w)
    left_tbl.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    left_x = doc.leftMargin + logo_w + 0.5 * cm
    left_tbl.wrapOn(canvas, doc.width, doc.topMargin)
    left_tbl.drawOn(canvas, left_x, info_y)

    # Right small table (ref / rev / page)
    right_data = [
        ['Q37440-00-EN-REG-00001'],
        ['REV A'],
        [f'Page {canvas.getPageNumber()}']
    ]
    right_col_w = 3 * cm
    right_tbl = Table(right_data, colWidths=[right_col_w])
    right_tbl.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    right_x = page_w - doc.rightMargin - right_col_w
    right_tbl.wrapOn(canvas, doc.width, doc.topMargin)
    right_tbl.drawOn(canvas, right_x, info_y)

    canvas.restoreState()


def generate_legal_register_first_page(PDF_PATH="legal_register_cover_page.pdf"):
    LOGO_PATH = str(logo_path)
    PAGE_WIDTH, PAGE_HEIGHT = A4

    # Ensure dummy logo creation uses a proper color string for PIL
    if not os.path.exists(LOGO_PATH):
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            img = PILImage.new('RGB', (240, 70), color='white')
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arialbd.ttf", 50)
            except Exception:
                font = ImageFont.load_default()
            # use a hex string for the fill (PIL expects color like "#RRGGBB")
            d.text((10, 5), "JESA", fill="#1F497D", font=font)
            os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
            img.save(LOGO_PATH)
        except Exception as e:
            # don't crash — warn and continue (we'll fallback to text)
            print(f"Warning: Could not create a dummy logo ({e}). Please provide {LOGO_PATH}.")

    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                            rightMargin=0.5 * inch, leftMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.75 * inch)
    elements = []
    content_width = PAGE_WIDTH - doc.leftMargin - doc.rightMargin

    header_style = ParagraphStyle(name="Header", fontName="Helvetica-Bold", fontSize=12, textColor=colors.white, leading=16, leftIndent=10)
    subheader_style = ParagraphStyle(name="Subheader", fontName="Helvetica", fontSize=7, textColor=colors.black, leading=12, spaceAfter=6)
    label_style = ParagraphStyle(name="Label", fontName="Helvetica-Bold", fontSize=6, textColor=colors.white, leftIndent=4, leading=12)
    value_style = ParagraphStyle(name="Value", fontName="Helvetica", fontSize=6, textColor=colors.black, leading=12)
    table_style = ParagraphStyle(name="TableText", fontName="Helvetica", fontSize=8, textColor=colors.black, leading=10, alignment=1)
    table_header_style = ParagraphStyle(name="TableHeader", fontName="Helvetica-Bold", fontSize=8, textColor=colors.white, leading=10, alignment=1)

    title_para = Paragraph("Sustainable Project Delivery - Legal Register - Chemical additives plant", header_style)

    # Safe creation of image: if not available, use a Spacer
    try:
        if os.path.exists(LOGO_PATH):
            logo_img = Image(LOGO_PATH, width=80, height=25)
        else:
            logo_img = Spacer(80, 25)
    except Exception:
        logo_img = Spacer(80, 25)

    purpose_para = Paragraph("Purpose of this register is to record the regulatory requirements that need to be complied with by the project. The register provides traceability of the action that has been taken to address the requirement.", subheader_style)
    combined_header_table = Table([[title_para, logo_img], [purpose_para, None]], colWidths=[content_width - 88, 88], rowHeights=[12 * mm, None])
    combined_header_table.setStyle(TableStyle([
        ('SPAN', (0, 1), (1, 1)),
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#1F497D")),
        ('BACKGROUND', (1, 0), (1, 0), colors.white),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBEFORE', (1, 0), (1, 0), 1, colors.black),
        ('TOPPADDING', (0, 1), (-1, 1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 5),
        ('LEFTPADDING', (0, 1), (-1, 1), 5),
        ('RIGHTPADDING', (0, 1), (-1, 1), 5),
    ]))
    elements.append(combined_header_table)
    elements.append(Spacer(1, 8 * mm))

    fields = [("PROJECT No:", "Q37440"), ("PROJECT TITLE:", "Chemical additives plant"), ("JESA DOCUMENT No:", "Q37440-00-EN-REG-00001"), ("ELECTRONIC FILE LOCATION:", "N/A"), ("NOTES:", "N/A")]
    for label, val in fields:
        label_para = Paragraph(label, label_style)
        label_table = Table([[label_para]], colWidths=[content_width], rowHeights=[7 * mm])
        label_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1F497D")), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        elements.append(label_table)
        elements.append(Spacer(1, 1.5 * mm))
        value_para = Paragraph(val, value_style)
        value_box = Table([[value_para]], colWidths=[content_width / 2])
        value_box.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.5, colors.grey), ('LEFTPADDING', (0, 0), (-1, -1), 4), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
        elements.append(value_box)
        elements.append(Spacer(1, 5 * mm))

    elements.append(Spacer(1, 40 * mm))
    originator_table_data = [[Paragraph("Originator:", table_style), Paragraph("Y.Hosni", table_style), Paragraph("Issue Date:", table_style), Paragraph("10-Aug-25", table_style)]]
    originator_table = Table(originator_table_data, colWidths=[content_width * 0.15, content_width * 0.35, content_width * 0.15, content_width * 0.35])
    originator_table.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.black), ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    elements.append(originator_table)

    status_header_data = [[Paragraph("DOCUMENT STATUS", table_header_style)]]
    status_header_table = Table(status_header_data, colWidths=[content_width])
    status_header_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1F497D")), ('BOX', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    elements.append(status_header_table)

    status_table_data = [
        [Paragraph("B", table_style), Paragraph("10-Aug-25", table_style), Paragraph("Issued for Review (IFR)", table_style), Paragraph("Y.Hosni", table_style), Paragraph("S.El Alem", table_style), Paragraph("J.Alaoui Sosse", table_style), Paragraph("S. Paresh", table_style)],
        [Paragraph("A", table_style), Paragraph("11-Mar-25", table_style), Paragraph("Issued for Internal Review (IIR)", table_style), Paragraph("I.Issa Issaka", table_style), Paragraph("S.El Alem", table_style), Paragraph("J.Alaoui Sosse", table_style), Paragraph("S. Salim", table_style)],
        [Paragraph("REV", table_style), Paragraph("DATE", table_style), Paragraph("DESCRIPTION", table_style), Paragraph("BY", table_style), Paragraph("CHKD", table_style), Paragraph("D.APPD", table_style), Paragraph("P.APPD", table_style)]
    ]
    status_table = Table(status_table_data, colWidths=[content_width * 0.06, content_width * 0.12, content_width * 0.35, content_width * 0.15, content_width * 0.12, content_width * 0.12, content_width * 0.08])
    num_rows = len(status_table_data)
    status_table.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.black), ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4), ('BACKGROUND', (0, num_rows - 1), (-1, num_rows - 1), colors.HexColor("#E6F3FF"))]))
    elements.append(status_table)

    copyright_para = Paragraph("© Copyright 2021 JESA Group...", ParagraphStyle(name="Copyright", fontName="Helvetica-Bold", fontSize=7, textColor=colors.black, leading=9, alignment=0, spaceAfter=0))
    elements.append(Spacer(1, 4 * mm))
    elements.append(copyright_para)

    doc.build(elements)


def generate_legal_register_story():
    story = []
    styles = getSampleStyleSheet()
    jesa_blue = colors.Color(red=0/255, green=51/255, blue=102/255)
    styles.add(ParagraphStyle(name='MainHeading', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=jesa_blue, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name='SubHeading', fontName='Helvetica-Bold', fontSize=7, leading=4, textColor=colors.black, spaceBefore=6, spaceAfter=0))
    body_style = styles['BodyText']; body_style.fontName = 'Helvetica'; body_style.fontSize = 7; body_style.leading = 10; body_style.alignment = TA_LEFT; body_style.spaceAfter = 4
    styles.add(ParagraphStyle(name='ListItem', parent=body_style, leftIndent=0, spaceBefore=0, spaceAfter=0))
    story.append(Paragraph("1. Objective", styles['MainHeading']))
    story.append(Paragraph("""The Legal Register is a listing of acts and regulations, including permits and approvals, applicable to the project or contract. The Legal Register is most useful on contracts and major projects where a significant number of regulations apply. The regulations can be sorted by
Project Phase, Discipline, or Activity, and thus more easily managed. The objective of using a Legal Register is to be proactive in managing regulatory requirements and therefore be within compliance for the duration of the project/contract. Normally, the Sustainable Project
Delivery (SPD) Lead will create and manage the Register, and provide updates to the Project Team on a regular basis. The SPD Lead is included in project activities such as status meetings to constantly inform relevant parties of their upcoming regulatory obligations on behalf of
the customer.
Creating and maintaining a Legal Register is agree between JESA and the Customer before project initiation and often during project development/Select phase.""", styles['BodyText']))
    story.append(Paragraph("2. Principals", styles['MainHeading']))
    for ref in ["• The Legal Register must be reviewed and approved by the Customer before use on a project or contract, as the Customer will own the Register.", "• The SPD Lead, or designate, shall work with the Discipline Leads and project manager(s) to populate and maintain an accurate status of each legal requirement on the register.", "• The SPD Lead, or designate, shall work with the Discipline Leads to determine which legal requirements must be incorporated into the design and who will be responsible for completing required actions.", "• The Legal Register can be used to create a Legal Matrix. A Legal Matrix is a table used to visually summarize the Legal Register. By listing project activities in a column and key legislation in a row across the top, markers can be placed to indicate all legislation that will impact a certain project activity."]:
        story.append(Paragraph(ref, styles['ListItem']))
    story.append(Paragraph("3. Instruction", styles['MainHeading']))
    story.append(Paragraph("Initiating the Regulatory Framework", styles['SubHeading']))
    story.append(Paragraph("The SPD Lead, or designate, shall develop the Legal Register as per the template or the needs of the project/contract.", styles['BodyText']))
    story.append(Paragraph("Identifying Regulatory Framework", styles['SubHeading']))
    story.append(Paragraph("A Legal Register is normally populated by a regulatory expert. The Customer and Project Manager shall assist with populating the register, and the Customer must approve the Register before use. The Register shall also be approved by the PM prior to use.", styles['BodyText']))
    story.append(Paragraph("Populating the Register", styles['SubHeading']))
    story.append(Paragraph("The framework headings will generally consist of:", styles['BodyText']))
    for ref in ["• ID","• Phase", "• Aspect / Activity", "• Impacts", "• Jurisdiction", "• Type", "• Regulatory Requirements"]:
        story.append(Paragraph(ref, styles['ListItem']))
    story.append(Paragraph("4. References", styles['MainHeading']))
    for ref in ["Safe and Sustainable Engineering for Asset Lifecycle (SEAL) Standard (MS-EP-STD-0017) Sustainable Project Delivery - Commitment Register (MS-EP-TEM-0051)", "Sustainable Solutions Standard (MS-PM-STD-0018)", "IFC – Guidelines Environmental, Health, and Safety Guidelines Environmental, Health and Safety Guidelines for Large Volume Inorganic Compounds Manufacturing and Coal Tar Distillation DECEMBER 10, 2007"]:
        story.append(Paragraph(ref, styles['ListItem']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("5. Abbreviations", styles['MainHeading']))
    usable_width = landscape(A3)[0] - 2.4 * inch
    final_table_data = [['ABH', 'Agence du Bassin Hydraulique', 'EHS', 'Environment, Health & Safety'], ['BAT', 'Best Available Technologies', 'HR', 'Human Resources'], ['CRI', 'Centre Régional d\'Investissement', 'IASE', 'Health Safety & Environment'], ['ONG', 'Organisation Non Gouvernementale', 'PSE', 'Programme de Suivi...'], ['OCP', 'Office Chérifien des Phosphates', 'SDG', 'Sustainable Development Goals'], ['', '', 'SEAL', 'Safe and Sustainable...']]
    abbrev_col_widths = [usable_width * 0.10, usable_width * 0.35, usable_width * 0.10, usable_width * 0.45]
    abbreviations_table = Table(final_table_data, colWidths=abbrev_col_widths)
    abbreviations_table.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 6), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('GRID', (0, 0), (-1, -1), 0.5, colors.black), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.append(abbreviations_table)
    return story


def create_legal_register_table_story(structured_data):
    pagesize = landscape(A3)
    margin = 1.0 * inch
    doc_width, doc_height = pagesize
    table_width = doc_width - 2 * margin

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(name='HeaderStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=5, leading=8, alignment=TA_CENTER, textColor=colors.white)
    category_header_style = ParagraphStyle(name='CategoryHeaderStyle', parent=header_style, fontSize=6, leading=10, alignment=TA_LEFT)
    cell_style = ParagraphStyle(name='CellStyle', parent=styles['Normal'], fontName='Helvetica', alignment=TA_LEFT, fontSize=5, leading=7)
    bold_cell_style = ParagraphStyle(name='BoldCellStyle', parent=cell_style, fontName='Helvetica-Bold')

    def create_header_paragraph(text): return Paragraph(str(text).replace('\n', '<br/>'), header_style)
    def create_category_paragraph(text): return Paragraph(str(text).replace('\n', '<br/>'), category_header_style)
    def create_paragraph(text, style=cell_style): return Paragraph(str(text).replace('\n', '<br/>'), style)

    header_row_str = ['Phase', 'Activity/Aspect', 'Impacts', 'Jurisdiction', 'Type', 'Legal Requirement', 'Date', 'Description', 'Task', 'Responsibility', 'Comments']
    table_data = [[create_header_paragraph(h) for h in header_row_str]]
    red_header_indices = []
    for category in structured_data:
        category_title = category.get("category_title", "Uncategorized")
        red_header_indices.append(len(table_data))
        red_header_row = [create_category_paragraph(category_title)] + [''] * (len(header_row_str) - 1)
        table_data.append(red_header_row)
        for row_data in category.get("rows", []):
            processed_row = [create_paragraph(row_data[0], bold_cell_style), *[create_paragraph(cell) for cell in row_data[1:]]]
            table_data.append(processed_row)

    col_widths_proportions = [0.07, 0.08, 0.07, 0.05, 0.05, 0.15, 0.06, 0.15, 0.12, 0.07, 0.13]
    col_widths = [p * table_width for p in col_widths_proportions]

    style_commands = [('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), 0.5, colors.black), ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002060')), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]
    for row_idx in red_header_indices:
        style_commands.append(('SPAN', (0, row_idx), (-1, row_idx)))
        style_commands.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#C00000')))

    table = Table(table_data, colWidths=col_widths)
    table.setStyle(TableStyle(style_commands))

    return [table]


def generate_combined_pdf_content(structured_data, output_filename):
    pagesize = landscape(A3)
    doc = SimpleDocTemplate(output_filename, pagesize=pagesize,
                            rightMargin=1.0 * inch, leftMargin=1.0 * inch,
                            topMargin=1.5 * inch, bottomMargin=1.5 * inch)

    story = generate_legal_register_story()
    table_story = create_legal_register_table_story(structured_data)

    # apply header/footer for first and later pages
    doc.build(story + [PageBreak()] + table_story, onFirstPage=header_footer, onLaterPages=header_footer)


def generate_complete_report_pdf(structured_data, final_output_path):
    """Create cover + content and merge to final_output_path (file path)."""
    temp_dir = tempfile.mkdtemp()
    try:
        cover_page_path = os.path.join(temp_dir, "cover_page.pdf")
        main_content_path = os.path.join(temp_dir, "main_content.pdf")

        # 1. cover
        generate_legal_register_first_page(cover_page_path)

        # 2. main content
        generate_combined_pdf_content(structured_data, main_content_path)

        # 3. merge
        merger = PdfMerger()
        merger.append(cover_page_path)
        merger.append(main_content_path)
        merger.write(final_output_path)
        merger.close()
        print(f"✅ Final merged PDF created: {final_output_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
