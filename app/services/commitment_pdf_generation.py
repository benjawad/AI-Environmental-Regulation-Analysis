from __future__ import annotations

import os
import io
import sys
import shutil
import logging
from typing import Optional, Any

# --- Django imports ---
try:
    from django.conf import settings
    from django.http import FileResponse, HttpResponse
except Exception:  # pragma: no cover - allow module import outside Django for linting
    settings = type("_S", (), {"MEDIA_ROOT": os.path.abspath("media"), "LOGO_PATH": os.path.abspath("media/jesa_logo.png")})
    class FileResponse:  # minimal shim for non-Django environments
        def __init__(self, *_, **__):
            raise RuntimeError("FileResponse used outside Django context")
    class HttpResponse:  # minimal shim for non-Django environments
        def __init__(self, *_, **__):
            raise RuntimeError("HttpResponse used outside Django context")

# --- Third-party imports ---
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.units import cm, inch, mm
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Flowable,
    Image,
    PageBreak,
)
from PyPDF2 import PdfMerger

# Optional PIL (used to create dummy logo)
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    PILImage = ImageDraw = ImageFont = None


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# --- Optional global expected by the user's pipeline ---
df_initial: Optional[pd.DataFrame] = globals().get("df_initial", None)


# ----------------------------
# Environment validation helper
# ----------------------------
def validate_parsing_setup() -> dict:
    """Validate runtime environment and return a summary dict.
    Mirrors the user's original intent to print validation, but also logs info.
    """
    info = {
        "media_root": getattr(settings, "MEDIA_ROOT", None),
        "logo_path": getattr(settings, "LOGO_PATH", None),
        "pandas_available": pd is not None,
        "reportlab_available": True,
        "pypdf2_available": True,
        "pil_available": PILImage is not None,
    }

    if not info["media_root"]:
        raise RuntimeError("Django setting MEDIA_ROOT is required for PDF generation.")

    os.makedirs(info["media_root"], exist_ok=True)

    if not info["logo_path"]:
        # define default logo under MEDIA_ROOT when not provided
        info["logo_path"] = os.path.join(info["media_root"], "jesa_logo.png")
        setattr(settings, "LOGO_PATH", info["logo_path"])  # set for downstream usage

    logger.info("Environment validation: %s", info)
    return info


# ------------------------------------
# Core function (entrypoint) - DO NOT rename
# ------------------------------------
def generate_commitment_register(request: Any):
    class VerticalText(Flowable):
        """A custom flowable to draw text rotated by 90 degrees."""
        def __init__(self, text, font_name='Helvetica-Bold', font_size=6):
            Flowable.__init__(self)
            self.text = text
            self.font_name = font_name
            self.font_size = font_size

        def draw(self):
            canvas = self.canv
            canvas.saveState()
            canvas.setFont(self.font_name, self.font_size)
            canvas.rotate(90)
            canvas.drawString(5, -self.font_size - 2, self.text)
            canvas.restoreState()

        def wrap(self, available_width, available_height):
            text_width = stringWidth(self.text, self.font_name, self.font_size)
            return (self.font_size + 4, text_width + 10)

    # --- HOW TO USE THIS FUNCTION ---
    validation = validate_parsing_setup()
    print("="*60)
    print("VALIDATION DE L'ENVIRONNEMENT")
    print("="*60)

    # -----------------
    # Header/footer impl
    # -----------------
    def header_footer(canvas, doc):
        jesa_blue = colors.Color(red=0/255, green=51/255, blue=102/255)

        header_offset = 0
        canvas.saveState()
        page_w, page_h = landscape(A3)

        # Shift the header downward by subtracting offset
        line_y = page_h - doc.topMargin + 2.15 * cm - header_offset
        canvas.setStrokeColor(jesa_blue)
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, line_y, page_w - doc.rightMargin, line_y)

        # --- Top separator line ---
        line_y = page_h - doc.topMargin + 0.5 * cm - header_offset
        canvas.setStrokeColor(jesa_blue)
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, line_y, page_w - doc.rightMargin, line_y)

        # --- Logo ---
        logo_path = getattr(settings, 'LOGO_PATH', None) or validation["logo_path"]
        logo_w = 4.0 * cm
        logo_h = 3.0 * cm
        logo_y = line_y + 0.2 * cm
        if logo_path and os.path.exists(logo_path):
            try:
                logo = Image(logo_path, width=logo_w, height=logo_h)
                logo.drawOn(canvas, doc.leftMargin, logo_y)
            except Exception:
                canvas.setFont('Helvetica-Bold', 30)
                canvas.setFillColor(jesa_blue)
                canvas.drawString(doc.leftMargin, logo_y + 0.4*cm, "JESA")
        else:
            canvas.setFont('Helvetica-Bold', 30)
            canvas.setFillColor(jesa_blue)
            canvas.drawString(doc.leftMargin, logo_y + 0.4*cm, "JESA")

        # --- Left table ---
        info_y = line_y - 0.2 * cm
        left_data = [
            ['Project Name:', 'Chemical additives plant'],
            ['Customer:',     'NOVADDIX'],
            ['Document Title:', 'Sustainable Project Delivery - Legal Register - Chemical additives plant']
        ]
        left_col_w = [(doc.width - logo_w - 0.5*cm)*0.1,
                      (doc.width - logo_w - 0.5*cm)*0.8]

        left_tbl = Table(left_data, colWidths=left_col_w)
        left_tbl.setStyle(TableStyle([
            ('FONTNAME',    (0,0), (0,-1),   'Helvetica-Bold'),
            ('FONTNAME',    (1,0), (1,-1),   'Helvetica'),
            ('FONTSIZE',    (0,0), (-1,-1),  6),
            ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING',(0,0), (-1,-1), 0),
        ]))
        left_x = doc.leftMargin + logo_w + 0.5 * cm
        left_tbl.wrapOn(canvas, doc.width, doc.topMargin)
        left_tbl.drawOn(canvas, left_x, info_y)

        # --- Right table ---
        right_data = [
            ['Q37440-00-EN-REG-00001'],
            ['REV A'],
            ['Page %d' % canvas.getPageNumber()]
        ]
        right_col_w = 3 * cm
        right_tbl = Table(right_data, colWidths=[right_col_w])
        right_tbl.setStyle(TableStyle([
            ('FONTNAME',    (0,0), (-1,-1),   'Helvetica-Bold'),
            ('FONTNAME',    (0,1), (0,1),     'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1),    6),
            ('ALIGN',       (0,0), (-1,-1), 'RIGHT'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING',(0,0), (-1,-1), 0),
        ]))
        right_x = page_w - doc.rightMargin - right_col_w
        right_tbl.wrapOn(canvas, doc.width, doc.topMargin)
        right_tbl.drawOn(canvas, right_x, info_y)

        canvas.restoreState()

    # ----------------
    # Second page body
    # ----------------
    def generate_commitment_register_second(output_filename):
        page_width, page_height = landscape(A3)

        # Setup document with balanced side margins
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=landscape(A3),
            rightMargin=1.2 * inch,
            leftMargin = 1.2* inch,
            topMargin=2.25 * inch,
            bottomMargin=0.75 * inch,
        )

        # Use full content width for tables
        usable_width = doc.width

        story = []
        styles = getSampleStyleSheet()

        # Custom color for JESA blue
        jesa_blue = colors.Color(red=0/255, green=51/255, blue=102/255)

        # Main heading style
        styles.add(ParagraphStyle(
            name='MainHeading',
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            textColor=jesa_blue,
            spaceBefore=8,
            spaceAfter=4
        ))

        # Sub-heading style
        styles.add(ParagraphStyle(
            name='SubHeading',
            fontName='Helvetica-Bold',
            fontSize=7,
            leading=4,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=0
        ))

        # Body text style
        body_style = styles['BodyText']
        body_style.fontName = 'Helvetica'
        body_style.fontSize = 7
        body_style.leading = 10
        body_style.alignment = 4  # Justified
        body_style.spaceAfter = 4

        # List item style
        styles.add(ParagraphStyle(
            name='ListItem',
            parent=body_style,
            leftIndent=0 * inch,
            spaceBefore=0,
            spaceAfter=0
        ))

        # --- Build document content ---
        story.append(Paragraph("1. Main purpose", styles['MainHeading']))
        story.append(Paragraph(
            """The Commitment Register is a system used to ensure commitments are incorporated into the appropriate part of engineering design, construction, procurement and/or operations, as required. Each commitment will be \"closed out\" in the Register before project phase completion, indicating that the commitment has been responsibly managed. A final Commitment Report is provided to the Customer at project phase completion outlining the inclusion of commitments into the various project documents and which commitments are compliant.""",
            styles['BodyText']
        ))

        story.append(Paragraph("2. Definition", styles['MainHeading']))
        story.append(Paragraph(
            "An obligation is a requirement, under the law, necessary for compliance. Obligations and compliance are managed as part of Technical Integrity under SEAl. A commitment is a voluntary statement of action, or a goal, that goes beyond legal requirements. The Commitment Register for a project or contract lists the commitments made by the Customer in corporate or publicly available documentation. Typical sources include the Environmental Impact Assessment (EIA), Project Registers/Application or material published for the public in newspapers, open houses, etc.",
            styles['BodyText']
        ))
        story.append(Paragraph(
            "The Commitment Register is a central place to document, communicate, and track the commitments so they will be understood and included in the project. This Commitment Register is part of SEAl Sustainable Design Planning, which is described in the SEAl Standard (MS-E9-STD-00017). The Commitment Register should be discussed with the Customer before use on a project or contract as part of SEAl Alignment, including how commitments are to be recorded and managed while executing a project.",
            styles['BodyText']
        ))
        story.append(Paragraph(
            "As the project progresses, commitments may become obsolete or may not be feasible to implement within the project. The Commitment Register is used to track the status of all commitments including rationale for those commitments that become obsolete or are not feasible. These changes in status are tracked in the Commitment Register.",
            styles['BodyText']
        ))

        story.append(Paragraph("3. Initiation", styles['MainHeading']))
        story.append(Paragraph("Initiating and Customizing the Commitment Register", styles['SubHeading']))
        story.append(Paragraph("The Project Manager / Project Engineering Manager or designate, shall:", styles['BodyText']))
        story.append(Paragraph("- work with the Customer to populate the Register and classify the commitments.", styles['ListItem']))
        story.append(Paragraph("- be responsible for ensuring commitments are registered and communicated to the appropriate party (e.g. the discipline lead responsible for incorporating a given commitment within the project scope of work).", styles['ListItem']))
        story.append(Paragraph("The Commitment Register is designed to be customizable to suit the project's commitment tracking needs. Columns such as 'Affected areas or processes' should be customized to reflect the project.", styles['BodyText']))

        story.append(Paragraph("Register Maintenance", styles['SubHeading']))
        story.append(Paragraph("The Project Manager, Project Engineering Manager or designate, shall work with the Discipline Leads to maintain an accurate status of each commitment on the register. The register shall be updated as needed and controlled properly so only the most recent version is available to the project team. Sufficient hours shall be included in the project budget for register maintenance.", styles['BodyText']))
        story.append(Paragraph("Technical Review", styles['SubHeading']))
        story.append(Paragraph("The Commitment Register shall be reviewed by the Project Management Team and approved by the Customer at an agreed frequency for the project. After each review and approval the signed Commitment Register shall be converted to PDF and saved while updates continue in the live register.", styles['BodyText']))
        story.append(Paragraph("Other Considerations", styles['SubHeading']))
        story.append(Paragraph("The commitments and other registers (Legal, Sustainable Solutions Database), are normally created in conjunction with the Sustainability Steering Committee (SSC). The SSC is comprised of sustainability stakeholders from the customer (e.g., public relations, environmental advisors, regulatory contacts, operations manager) and JESA (e.g., Sustainability Lead, environmental scientists).", styles['BodyText']))

        story.append(Paragraph("4. References", styles['MainHeading']))
        for ref in [
            "Safe and Sustainable Engineering for Asset Lifecycle (SEAL) Standard (MS-E9-STD-00017)",
            "Sustainable Project Delivery - Legal Register (MS-E9-TEM-00053)",
            "Sustainable Solutions Standard (MS-FM-STD-00158)",
        ]:
            story.append(Paragraph(ref, styles['ListItem']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("5. Abbreviations", styles['MainHeading']))
        final_table_data = [
            ['ABH', 'Agence du Bassin Hydraulique', 'EHS', 'Environment, Health & Safety'],
            ['BAT', 'Best Available Technologies', 'HR', 'Human Resources'],
            ['CRI', "Centre Régional d'Investissement", 'IASE', 'Health Safety & Environment'],
            ['ONG', 'Organisation Non Gouvernementale', 'PSE', 'Programme de Suivi et de Surveillance Environnemental'],
            ['OCP', 'Office Chérifien des Phosphates', 'SDG', 'Sustainable Development Goals'],
            ['', '', 'SEAL', 'Safe and Sustainable Engineering for Asset Lifecycle'],
        ]
        abbrev_col_widths = [usable_width * 0.10, usable_width * 0.35, usable_width * 0.10, usable_width * 0.45]
        abbreviations_table = Table(final_table_data, colWidths=abbrev_col_widths)
        abbreviations_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(abbreviations_table)
        return story

    # --------------
    # Cover page impl
    # --------------
    def generate_commitment_register_cover_page (PDF_PATH = "commitment_register_cover_page.pdf"):
        LOGO_PATH = getattr(settings, 'LOGO_PATH', None)
        PAGE_WIDTH, PAGE_HEIGHT = A4

        # Ensure parent directory exists if a full path was provided
        if PDF_PATH:
            os.makedirs(os.path.dirname(os.path.abspath(PDF_PATH)) or '.', exist_ok=True)

        # --- Create a Dummy Logo if it doesn't exist ---
        if not LOGO_PATH:
            LOGO_PATH = os.path.join(getattr(settings, 'MEDIA_ROOT', '.'), 'jesa_logo.png')
            setattr(settings, 'LOGO_PATH', LOGO_PATH)
        if not os.path.exists(LOGO_PATH) and PILImage is not None:
            try:
                img = PILImage.new('RGB', (240, 70), color='white')
                d = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arialbd.ttf", 50)
                except Exception:
                    font = ImageFont.load_default()
                # Fallback to black if colors.HexColor is not acceptable here
                d.text((10, 5), "JESA", fill=(31, 73, 125), font=font)
                os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
                img.save(LOGO_PATH)
                logger.info(f"Created a dummy logo: {LOGO_PATH}")
            except Exception as e:
                logger.warning(f"Could not create a dummy logo. Please provide {LOGO_PATH}. Error: {e}")

        # --- Document Setup ---
        doc = SimpleDocTemplate(
                PDF_PATH,
                pagesize=A4,
                rightMargin=0.5 * inch,
                leftMargin = 0.5* inch,
                topMargin=0.5 * inch,
                bottomMargin=0.75 * inch,
            )
        elements = []
        content_width = PAGE_WIDTH - doc.leftMargin - doc.rightMargin

        # --- Define Paragraph Styles ---
        header_style = ParagraphStyle(
            name="Header",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.white,
            leading=16,
            leftIndent=10
        )
        subheader_style = ParagraphStyle(
            name="Subheader",
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.black,
            leading=12,
            spaceAfter=6,
        )
        label_style = ParagraphStyle(
            name="Label",
            fontName="Helvetica-Bold",
            fontSize=6,
            textColor=colors.white,
            leftIndent=4,
            leading=12,
        )
        value_style = ParagraphStyle(
            name="Value",
            fontName="Helvetica-bold",
            fontSize=6,
            textColor=colors.black,
            leading=12,
        )
        table_style = ParagraphStyle(
            name="TableText",
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.black,
            leading=10,
            alignment=1,
        )
        table_header_style = ParagraphStyle(
            name="TableHeader",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.white,
            leading=10,
            alignment=1,
        )

        # --- Combined Header and Purpose Box ---
        title_para = Paragraph(
            "Sustainable Project Delivery - Legal Register - Chemical additives plant",
            header_style
        )
        logo_img_path = getattr(settings, 'LOGO_PATH', None)
        if logo_img_path and os.path.exists(logo_img_path):
            logo_img = Image(logo_img_path, width=80, height=25)
        else:  # graceful fallback
            logo_img = Table([[Paragraph("JESA", header_style)]], colWidths=[88])

        purpose_para = Paragraph(
            "Purpose of this register is to record the regulatory requirements that need to be complied with by the project. "
            "The register provides traceability of the action that has been taken to address the requirement.",
            subheader_style
        )

        combined_header_table = Table(
            [
                [title_para, logo_img],
                [purpose_para, None]
            ],
            colWidths=[content_width - 88, 88],
            rowHeights=[12*mm, None]
        )
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
        elements.append(Spacer(1, 8*mm))

        # --- Project Detail Fields ---
        fields = [
            ("PROJECT No:", "Q37440"),
            ("PROJECT TITLE:", "Chemical additives plant"),
            ("JESA DOCUMENT No:", "Q37440-00-EN-REG-00001"),
            ("ELECTRONIC FILE LOCATION:", "N/A"),
            ("NOTES:", "N/A"),
        ]

        for label, val in fields:
            label_para = Paragraph(label, label_style)
            label_table = Table([[label_para]], colWidths=[content_width], rowHeights=[7*mm])
            label_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1F497D")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(label_table)
            elements.append(Spacer(1, 1.5*mm))

            value_para = Paragraph(val, value_style)
            value_box = Table([[value_para]], colWidths=[content_width / 2])
            value_box.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(value_box)
            elements.append(Spacer(1, 5*mm))

        # --- Bottom Status Table ---
        elements.append(Spacer(1, 40*mm))

        originator_table_data = [[
            Paragraph("Originator:", table_style),
            Paragraph("Y.Hosni", table_style),
            Paragraph("Issue Date:", table_style),
            Paragraph("18-Jun-25", table_style),
        ]]
        originator_table = Table(
            originator_table_data,
            colWidths=[content_width * 0.15, content_width * 0.35, content_width * 0.15, content_width * 0.35]
        )
        originator_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(originator_table)

        status_header_table = Table([[Paragraph("DOCUMENT STATUS", table_header_style)]], colWidths=[content_width])
        status_header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1F497D")),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(status_header_table)

        status_table_data = [
            [Paragraph("B", table_style), Paragraph("18-Jun-25", table_style), Paragraph("Issued for Review (IFR)", table_style), Paragraph("Y.Hosni", table_style), Paragraph("S.El Alem", table_style), Paragraph("J.Alaoui Sosse", table_style), Paragraph("S. Paresh", table_style)],
            [Paragraph("A", table_style), Paragraph("11-Mar-25", table_style), Paragraph("Issued for Internal Review (IIR)", table_style), Paragraph("I.Issa Issaka", table_style), Paragraph("S.El Alem", table_style), Paragraph("J.Alaoui Sosse", table_style), Paragraph("S. Salim", table_style)],
            [Paragraph("REV", table_style), Paragraph("DATE", table_style), Paragraph("DESCRIPTION", table_style), Paragraph("BY", table_style), Paragraph("CHKD", table_style), Paragraph("D.APPD", table_style), Paragraph("P.APPD", table_style)],
        ]
        status_table = Table(
            status_table_data,
            colWidths=[
                content_width * 0.06,
                content_width * 0.12,
                content_width * 0.35,
                content_width * 0.15,
                content_width * 0.12,
                content_width * 0.12,
                content_width * 0.08,
            ]
        )
        num_rows = len(status_table_data)
        status_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, num_rows - 1), (-1, num_rows - 1), colors.HexColor("#E6F3FF")),
        ]))
        elements.append(status_table)

        copyright_para = Paragraph(
            "© Copyright 2021 JESA Group. No part of this document or the information it contains may be reproduced or transmitted in any form or by any means electronic or mechanical, including photocopying, recording, or by any information storage and retrieval system, without permission in writing from JESA. JESA.com",
            ParagraphStyle(
                name="Copyright",
                fontName="Helvetica-bold",
                fontSize=7,
                textColor=colors.black,
                leading=9,
                alignment=0,
                spaceAfter=0,
            )
        )
        elements.append(Spacer(1, 4*mm))
        elements.append(copyright_para)

        # --- Render the PDF ---
        doc.build(elements)
        logger.info(f"✅ PDF successfully created: {PDF_PATH}")
        print("✅ PDF de couverture créé : commitment_register_cover_page.pdf")

    # Immediately generate cover so downstream calls always have it
    generate_commitment_register_cover_page()

    # --------------------------------------------------
    # Final Commitment Register (tabular content from df)
    # --------------------------------------------------
    def generate_commitment_register_pdf(df, output_filename="commitment_register_final.pdf"):
        """
        Generates a PDF file with a detailed table dynamically from a DataFrame
        with a 3-level MultiIndex header.
        """
        pagesize = landscape(A3)
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=pagesize,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=1.3 * inch,
        )

        table_width = doc.width

        # --- Paragraph Styles for Cell Content ---
        styles = getSampleStyleSheet()
        cell_style = ParagraphStyle(
            name='CellStyle',
            parent=styles['Normal'],
            alignment=TA_CENTER,
            fontSize=6,
            leading=8,
            spaceAfter=2,
            spaceBefore=2
        )
        header_style = ParagraphStyle(
            name='HeaderStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7,
            leading=8,
            alignment=TA_CENTER,
            textColor=colors.white,
        )

        # --- Helper Functions ---
        def create_header_paragraph(text, style=header_style):
            return Paragraph(str(text).replace('\n', '<br/>'), style)

        def create_data_paragraph(text, style=cell_style):
            if not isinstance(text, str):
                text = str(text)
            return Paragraph(text.replace('\n', '<br/>').replace('* ', '• '), style)

        # --- DYNAMIC HEADER GENERATION ---
        header_row_1 = []
        header_row_2 = []
        header_row_3 = []
        last_h1 = None
        last_h2 = None

        # Validate df
        if pd is None:
            raise RuntimeError("pandas is required to build the commitment register table.")
        if not isinstance(df, pd.DataFrame) or not hasattr(df.columns, 'levels') or len(getattr(df.columns, 'levels', [])) < 3:
            raise ValueError("DataFrame must have a 3-level MultiIndex columns structure.")

        for l0, l1, l2 in df.columns:
            if l0 != last_h1:
                header_row_1.append(create_header_paragraph(l0))
                last_h1 = l0
            else:
                header_row_1.append('')
            if l1 != last_h2:
                header_row_2.append(create_header_paragraph(l1))
                last_h2 = l1
            else:
                header_row_2.append('')
            header_row_3.append(VerticalText(l2) if l2 else '')

        # --- DYNAMIC DATA ROW GENERATION ---
        data_rows = []
        for _, row in df.iterrows():
            pdf_row = [create_data_paragraph(cell) for cell in row]
            data_rows.append(pdf_row)

        table_data = [header_row_1, header_row_2, header_row_3] + data_rows

        # --- Dynamic span styles across multi-level headers ---
        dynamic_styles = []
        level0_headers = df.columns.get_level_values(0)
        start_col = 0
        for i in range(1, len(level0_headers)):
            if level0_headers[i] != level0_headers[i-1]:
                dynamic_styles.append(('SPAN', (start_col, 0), (i - 1, 0)))
                start_col = i
        dynamic_styles.append(('SPAN', (start_col, 0), (len(level0_headers) - 1, 0)))

        level1_headers = df.columns.droplevel(2)
        start_col = 0
        for i in range(1, len(level1_headers)):
            if level1_headers[i] != level1_headers[i-1]:
                dynamic_styles.append(('SPAN', (start_col, 1), (i - 1, 1)))
                start_col = i
        dynamic_styles.append(('SPAN', (start_col, 1), (len(level1_headers) - 1, 1)))

        for i, (l0, l1, l2) in enumerate(df.columns):
            if not l1 and not l2:  # simple, non-nested column
                dynamic_styles.append(('SPAN', (i, 0), (i, 2)))
            elif l1 and not l2:  # spans row 1 and 2
                dynamic_styles.append(('SPAN', (i, 1), (i, 2)))

        # Column widths (proportional -> absolute)
        col_widths_proportions = [
            0.05, 0.05, 0.04, 0.12, 0.04, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04, 0.08,
            0.06, 0.06, 0.04,
            0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015,  # Affected Areas
            0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012,  # Impact
            0.06, 0.033
        ]
        table_width = doc.width
        col_widths = [p * table_width for p in col_widths_proportions]

        final_style = TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002060')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, 2), colors.HexColor('#C00000')),
            ('TEXTCOLOR', (0, 1), (-1, 2), colors.white),
            ('FONTNAME', (0, 1), (-1, 2), 'Helvetica-Bold'),
        ] + dynamic_styles)

        table = Table(table_data, colWidths=col_widths, repeatRows=3)
        table.setStyle(final_style)
        story = generate_commitment_register_second(output_filename)
        doc.build(story + [PageBreak()] + [table], onFirstPage=header_footer, onLaterPages=header_footer)
        logger.info(f"PDF successfully generated: {output_filename}")

    # ----------------------
    # Assembly / Merge (I/O)
    # ----------------------
    # Keep original name `df_final`
    global df_initial
    df_final = df_initial.copy() if (pd is not None and isinstance(df_initial, pd.DataFrame)) else None

    # Fallback: try request attribute without renaming variables
    if df_final is None and hasattr(request, 'df_initial'):
        df_final = getattr(request, 'df_initial')

    if df_final is None:
        logger.error("df_initial is not provided. Ensure a 3-level MultiIndex DataFrame is available.")
        return HttpResponse("Erreur : df_initial manquant pour générer le PDF", status=400)

    def build_full_pdf(df_final, output_name="commitment_register.pdf"):
        # Chemin complet pour le répertoire media
        media_path = settings.MEDIA_ROOT
        os.makedirs(media_path, exist_ok=True)

        # Chemins complets pour tous les fichiers
        cover_path = os.path.join(media_path, "commitment_register_cover_page.pdf")
        content_path = os.path.join(media_path, "commitment_register_final.pdf")
        final_path = os.path.join(media_path, output_name)

        # 1. Générer la couverture dans le répertoire courant puis déplacer
        generate_commitment_register_cover_page()
        if os.path.exists("commitment_register_cover_page.pdf"):
            try:
                shutil.move("commitment_register_cover_page.pdf", cover_path)
            except Exception:
                # fallback copy+remove
                shutil.copy2("commitment_register_cover_page.pdf", cover_path)
                os.remove("commitment_register_cover_page.pdf")

        # 2. Générer le contenu principal puis déplacer
        generate_commitment_register_pdf(df_final, output_filename="commitment_register_final.pdf")
        if os.path.exists("commitment_register_final.pdf"):
            try:
                shutil.move("commitment_register_final.pdf", content_path)
            except Exception:
                shutil.copy2("commitment_register_final.pdf", content_path)
                os.remove("commitment_register_final.pdf")

        # 3. Fusionner les PDFs uniquement si les deux fichiers existent
        if os.path.exists(cover_path) and os.path.exists(content_path):
            try:
                merger = PdfMerger()
                with open(cover_path, 'rb') as f1:
                    merger.append(f1)
                with open(content_path, 'rb') as f2:
                    merger.append(f2)
                with open(final_path, 'wb') as fout:
                    merger.write(fout)
                merger.close()
            finally:
                # Supprimer les fichiers temporaires
                try:
                    os.remove(cover_path)
                except Exception:
                    pass
                try:
                    os.remove(content_path)
                except Exception:
                    pass

            logger.info(f"✅ PDF final créé : {final_path}")
            return final_path
        else:
            logger.error("❌ Erreur: Fichiers manquants pour la fusion")
            return None

    # Dans votre vue principale (kept as in the user's snippet)
    pdf_path = build_full_pdf(df_final)

    if pdf_path and os.path.exists(pdf_path):
        try:
            return FileResponse(open(pdf_path, 'rb'), as_attachment=True, filename='commitment_register.pdf')
        except Exception as e:
            logger.exception("Failed to return FileResponse: %s", e)
            return HttpResponse("Erreur lors de l'ouverture du fichier PDF", status=500)
    else:
        return HttpResponse("Erreur : le fichier PDF n'a pas pu être généré", status=500)
