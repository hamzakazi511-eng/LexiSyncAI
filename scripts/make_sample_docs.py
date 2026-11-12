"""Generate a synthetic, fictional 3-PDF legal bundle used for tests and the evaluation set.

The bundle deliberately contains the hard cases from the brief:
  * repeating headers/footers + page numbers   * a two-column definitions page
  * a footnote                                  * an indemnity table (Table 9.1)
  * a scanned, image-only signature page (OCR)  * a figure-only page (anomaly detector)
  * an amendment that conflicts with the original contract (cap EUR 2.0M -> EUR 2.5M)
All parties and figures are invented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "samples"
W, H = A4
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=17)
H1 = ParagraphStyle("h1", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=10)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=13.5, spaceAfter=5)
NOTE = ParagraphStyle("n", parent=BODY, fontSize=7.5, leading=9)


def _doc(path: Path, header: str, two_col_pages: bool = False) -> BaseDocTemplate:
    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, H - 1.2 * cm, header)
        canvas.drawRightString(W - 2 * cm, 1.0 * cm, f"Page {doc.page}")
        canvas.restoreState()

    d = BaseDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.2 * cm,
                        bottomMargin=2 * cm, title=header.split(" – ")[0])
    one = Frame(d.leftMargin, d.bottomMargin, d.width, d.height, id="one")
    gap = 0.8 * cm
    colw = (d.width - gap) / 2
    left = Frame(d.leftMargin, d.bottomMargin, colw, d.height, id="l")
    right = Frame(d.leftMargin + colw + gap, d.bottomMargin, colw, d.height, id="r")
    d.addPageTemplates([PageTemplate("one", [one], onPage=deco), PageTemplate("two", [left, right], onPage=deco)])
    return d


def P(t, s=BODY):
    return Paragraph(t, s)


# ------------------------------------------------------------------------------------------ MSA
def make_msa(path: Path):
    d = _doc(path, "Master Services Agreement – Nordwind Logistics GmbH / Clearpath Software Ltd")
    s = [P("Master Services Agreement", TITLE),
         P("This Master Services Agreement (the <b>Agreement</b>) is entered into on 15 January 2025 between "
           "<b>Nordwind Logistics GmbH</b>, Speicherstadt 12, 20457 Hamburg, Germany (the <b>Customer</b>) and "
           "<b>Clearpath Software Ltd</b>, 4 Harbour Row, Dublin 2, Ireland (the <b>Supplier</b>)."),
         P("The Supplier provides a cloud-based fleet routing and warehouse optimisation platform. The Customer "
           "wishes to use the platform across its European logistics network on the terms set out below."),
         P("2. SERVICES", H1),
         P("2.1 The Supplier shall provide the Services described in Schedule A, including hosting, maintenance, "
           "updates, second-level support and onboarding of up to 400 named users."),
         P("2.2 The Supplier shall perform the Services with reasonable skill and care and in accordance with good "
           "industry practice and all applicable laws."),
         P("3. TERM AND TERMINATION", H1),
         P("3.1 This Agreement commences on the Effective Date and continues for an initial term of thirty-six (36) "
           "months, after which it renews automatically for successive periods of twelve (12) months."),
         P("3.2 The Customer may terminate this Agreement for convenience by giving the Supplier not less than "
           "ninety (90) days' written notice."),
         P("3.3 Either party may terminate this Agreement with immediate effect if the other party commits a "
           "material breach which is not remedied within thirty (30) days of written notice."),
         NextPageTemplate("two"), PageBreak(),
         P("1. DEFINITIONS", H1),
         ]
    defs = [("Affiliate", "any entity that directly or indirectly controls, is controlled by, or is under common control with a party."),
            ("Business Day", "a day other than a Saturday, Sunday or public holiday in Hamburg, Germany."),
            ("Confidential Information", "all information disclosed by a party that is marked as confidential or would reasonably be regarded as confidential."),
            ("Customer Data", "all data, including personal data, uploaded to the platform by or on behalf of the Customer."),
            ("Effective Date", "1 February 2025."),
            ("Fees", "the fees payable by the Customer as set out in Section 4."),
            ("Personal Data", "has the meaning given in Article 4(1) of Regulation (EU) 2016/679 (GDPR)."),
            ("Service Credits", "the credits calculated in accordance with Section 5.3."),
            ("Services", "the services described in Schedule A, as amended from time to time."),
            ("Term", "the initial term together with any renewal term."),
            ("Data Processing Addendum", "the addendum agreed between the parties governing the processing of Personal Data."),
            ("Losses", "all losses, liabilities, damages, costs and reasonable legal fees.")]
    for k, v in defs:
        s.append(P(f"<b>“{k}”</b> means {v}"))
        s.append(P("The definitions in this Section apply throughout this Agreement, its Schedules and any "
                   "amendment, unless the context requires otherwise.") if k == "Fees" else Spacer(1, 2))
    s += [NextPageTemplate("one"), PageBreak(),
          P("4. FEES AND PAYMENT", H1),
          P("4.1 The Customer shall pay the Supplier an annual subscription fee of EUR 1,200,000 (one million two "
            "hundred thousand euros), invoiced quarterly in advance."),
          P("4.2 Invoices are payable within thirty (30) days of receipt. Late payments bear interest at eight "
            "percentage points above the base rate<super>1</super>."),
          P("4.3 Fees are exclusive of VAT, which shall be payable in addition at the applicable rate."),
          P("5. SERVICE LEVELS", H1),
          P("5.1 The Supplier shall ensure that the platform is available at least 99.5% of the time in each "
            "calendar month, excluding scheduled maintenance announced at least 5 Business Days in advance."),
          P("5.2 Critical incidents shall be responded to within one (1) hour and resolved or worked around within "
            "eight (8) hours."),
          P("5.3 If availability falls below the level in Section 5.1, the Customer is entitled to Service Credits "
            "of 2% of the monthly fee for each 0.5% shortfall, capped at 10% of the monthly fee."),
          P("6. CONFIDENTIALITY", H1),
          P("6.1 Each party shall keep the other party's Confidential Information confidential and use it only to "
            "perform its obligations under this Agreement."),
          P("6.2 The obligations in this Section 6 survive termination of this Agreement for a period of five (5) "
            "years."),
          Spacer(1, 150),
          P("1 The base rate published by the Deutsche Bundesbank pursuant to Section 247 of the German Civil Code "
            "(BGB).", NOTE),
          PageBreak(),
          P("7. LIMITATION OF LIABILITY", H1),
          P("7.1 Neither party shall be liable for any indirect or consequential loss, loss of profit, loss of "
            "revenue or loss of goodwill."),
          P("7.2 Subject to Section 7.3, each party's total aggregate liability arising out of or in connection with "
            "this Agreement shall not exceed EUR 2,000,000 (two million euros) in any contract year."),
          P("7.3 Nothing in this Agreement limits liability for death or personal injury caused by negligence, for "
            "fraud, or for wilful misconduct."),
          P("8. DATA PROTECTION", H1),
          P("8.1 The parties shall comply with the Data Processing Addendum, which forms part of this Agreement. In "
            "the event of conflict regarding Personal Data, the Data Processing Addendum prevails."),
          P("9. INDEMNITIES", H1),
          P("9.1 The Supplier shall indemnify the Customer against all Losses arising from the claims listed in "
            "Table 9.1, up to the amounts stated for each indemnity. These indemnities are not subject to the cap "
            "in Section 7.2."),
          P("Table 9.1 – Supplier indemnities"),
          ]
    t = Table([["Indemnity", "Trigger", "Maximum amount"],
               ["Intellectual property", "Third-party IP infringement claim", "EUR 2,000,000"],
               ["Data breach", "Breach of the Data Processing Addendum", "EUR 1,000,000"],
               ["Personal injury / property", "Third-party injury or damage on Customer sites", "EUR 500,000"],
               ["Total", "", "EUR 3,500,000"]], colWidths=[4.5 * cm, 7 * cm, 4 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    s += [t, Spacer(1, 8),
          P("9.2 The indemnified party shall notify the indemnifying party of any claim within fourteen (14) days and "
            "allow it to conduct the defence."),
          P("10. INSURANCE", H1),
          P("10.1 The Supplier shall maintain professional indemnity insurance with a limit of not less than "
            "EUR 5,000,000 per claim and cyber liability insurance of not less than EUR 3,000,000."),
          PageBreak(),
          P("11. FORCE MAJEURE", H1),
          P("11.1 Neither party is liable for delay caused by events beyond its reasonable control. If such an event "
            "continues for more than sixty (60) days, either party may terminate on written notice."),
          P("12. GOVERNING LAW AND JURISDICTION", H1),
          P("12.1 This Agreement and any non-contractual obligations arising out of it are governed by the laws of "
            "the Federal Republic of Germany, excluding the UN Convention on Contracts for the International Sale "
            "of Goods."),
          P("12.2 The courts of Hamburg, Germany have exclusive jurisdiction."),
          Spacer(1, 30),
          P("Signed for Nordwind Logistics GmbH: Jonas Petersen, Managing Director"),
          P("Signed for Clearpath Software Ltd: Maria Jensen, Chief Executive Officer")]
    d.build(s)


# ------------------------------------------------------------------------------------ Amendment
def make_amendment(path: Path):
    d = _doc(path, "Amendment No. 1 – Master Services Agreement Nordwind / Clearpath")
    s = [P("Amendment No. 1 to the Master Services Agreement", TITLE),
         P("This Amendment No. 1 is made on 14 February 2026 between Nordwind Logistics GmbH (the Customer) and "
           "Clearpath Software Ltd (the Supplier) and amends the Master Services Agreement dated 15 January 2025 "
           "(the Agreement). It takes effect on 1 March 2026."),
         P("1. BACKGROUND", H1),
         P("1.1 The Customer is expanding the platform to its Scandinavian warehouses and the parties wish to adjust "
           "the commercial terms accordingly."),
         P("2. AMENDMENTS", H1),
         P("2.1 Section 4.1 of the Agreement is amended so that the annual subscription fee is EUR 1,350,000 (one "
           "million three hundred and fifty thousand euros) with effect from 1 March 2026."),
         P("2.2 Section 7.2 of the Agreement is amended by replacing “EUR 2,000,000 (two million euros)” with "
           "“EUR 2,500,000 (two million five hundred thousand euros)”."),
         P("2.3 Section 3.1 of the Agreement is amended so that the initial term is forty-eight (48) months from the "
           "Effective Date."),
         P("2.4 The maximum number of named users in Schedule A is increased from 400 to 650."),
         P("3. GENERAL", H1),
         P("3.1 Except as amended by this Amendment, the Agreement remains in full force and effect. In case of any "
           "conflict between this Amendment and the Agreement, this Amendment prevails."),
         P("3.2 Section 12 of the Agreement (Governing Law and Jurisdiction) applies to this Amendment."),
         PageBreak(),
         P("Schedule C – Platform availability 2025", H1)]
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing
    dr = Drawing(16 * cm, 9 * cm)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 40, 30, 14 * cm, 7 * cm
    bc.data = [[99.7, 99.8, 99.2, 99.9, 99.6, 99.9, 99.8, 98.9, 99.7, 99.9, 99.8, 99.9]]
    bc.valueAxis.valueMin, bc.valueAxis.valueMax = 98.5, 100
    bc.categoryAxis.categoryNames = list("JFMAMJJASOND")
    bc.bars[0].fillColor = colors.steelblue
    dr.add(bc)
    s.append(dr)
    d.build(s)

    # append a scanned (image-only) signature page
    img = Image.new("L", (1240, 1754), 250)
    dw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
    except OSError:
        font = big = ImageFont.load_default()
    lines = [("Schedule B - Signatories", big), ("", font),
             ("Signed for and on behalf of Clearpath Software Ltd", font),
             ("by Maria Jensen, Chief Executive Officer,", font), ("on 14 February 2026 in Dublin.", font), ("", font),
             ("Signed for and on behalf of Nordwind Logistics GmbH", font),
             ("by Jonas Petersen, Managing Director,", font), ("on 16 February 2026 in Hamburg.", font)]
    y = 220
    for text, f in lines:
        dw.text((150, y), text, fill=20, font=f)
        y += 62
    img = img.rotate(0.6, fillcolor=250).filter(ImageFilter.GaussianBlur(0.6))
    tmp = path.with_suffix(".scan.png")
    img.save(tmp)
    pdf = pymupdf.open(path)
    page = pdf.new_page(width=W, height=H)
    page.insert_image(page.rect, filename=str(tmp))
    pdf.save(str(path) + ".tmp")
    pdf.close()
    Path(str(path) + ".tmp").replace(path)
    tmp.unlink()


# ------------------------------------------------------------------------------------------ DPA
def make_dpa(path: Path):
    d = _doc(path, "Data Processing Addendum – Nordwind Logistics GmbH / Clearpath Software Ltd")
    s = [P("Data Processing Addendum", TITLE),
         P("This Data Processing Addendum forms part of the Master Services Agreement between Nordwind Logistics "
           "GmbH (Controller) and Clearpath Software Ltd (Processor) and reflects the requirements of Article 28 "
           "of Regulation (EU) 2016/679 (GDPR)."),
         P("Article 1", H1), P("Subject matter and duration"),
         P("1. The Processor processes Personal Data on behalf of the Controller solely for the provision of the "
           "Services for the duration of the Agreement."),
         P("2. The categories of data subjects are employees, drivers and customers' delivery contacts of the "
           "Controller. The data categories are contact details, location data and delivery records."),
         P("Article 2", H1), P("Instructions"),
         P("1. The Processor shall process Personal Data only on documented instructions from the Controller."),
         P("2. The Processor shall immediately inform the Controller if, in its opinion, an instruction infringes "
           "the GDPR or other Union or Member State data protection provisions."),
         P("Article 3", H1), P("Sub-processors"),
         P("1. The Controller authorises the following sub-processors: Amazon Web Services EMEA SARL (hosting, "
           "Frankfurt region) and Zendesk Inc. (support ticketing)."),
         P("2. The Processor shall inform the Controller of any intended addition or replacement of sub-processors "
           "at least thirty (30) days in advance, giving the Controller the opportunity to object."),
         P("Article 4", H1), P("Personal data breaches"),
         P("1. The Processor shall notify the Controller without undue delay and in any event within forty-eight "
           "(48) hours after becoming aware of a personal data breach."),
         P("2. The notification shall describe the nature of the breach, the categories and approximate number of "
           "data subjects concerned, the likely consequences and the measures taken or proposed."),
         PageBreak(),
         P("Article 5", H1), P("Deletion and return of data"),
         P("1. Upon termination of the Agreement, the Processor shall, at the choice of the Controller, return all "
           "Personal Data and delete existing copies within sixty (60) days, unless Union or Member State law "
           "requires storage."),
         P("Article 6", H1), P("Audits"),
         P("1. The Controller may audit the Processor's compliance with this Addendum once per calendar year, with "
           "at least thirty (30) days' prior written notice."),
         P("Article 7", H1), P("International transfers"),
         P("1. Personal Data shall not be transferred outside the European Economic Area except under the Standard "
           "Contractual Clauses adopted by the European Commission in Decision (EU) 2021/914."),
         ]
    d.build(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_msa(OUT / "01_master_services_agreement.pdf")
    make_amendment(OUT / "02_amendment_no1.pdf")
    make_dpa(OUT / "03_data_processing_addendum.pdf")
    print("written to", OUT)


if __name__ == "__main__":
    main()
