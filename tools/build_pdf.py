"""Compile the study guide: python tools/build_pdf.py  ->  reports/ML_Fundamentals_Study_Guide.pdf"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdflib import *
import c00_front, c01_ch1, c02_ch2, c03_ch3, c04_ch4, c05_ch5, c06_ch6, c07_ch7, c08_ch8, c09_ch9, c10_final, c11_appendix

out = ROOT / "reports" / "ML_Fundamentals_Study_Guide.pdf"
story = []
story += c00_front.cover()
story += [Paragraph("Contents", S["h1c"]), toc(), PageBreak()]
story += c00_front.how_to_use()
story += c00_front.project_brief()
story += c00_front.roadmap()
for m in (c01_ch1, c02_ch2, c03_ch3, c04_ch4, c05_ch5, c06_ch6, c07_ch7, c08_ch8, c09_ch9, c10_final):
    story += m.build()
story += c11_appendix.build()
doc = Doc(str(out))
doc.multiBuild(story)
print("built", out, out.stat().st_size // 1024, "KB")
