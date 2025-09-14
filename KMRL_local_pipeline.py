import os, io, json
from pathlib import Path
from collections import defaultdict

from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader, PdfWriter

from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- CONFIG ----------
INPUT_PDF = "sample.pdf"   # put your multi-department file here
OUTPUT_DIR = "KMRL_Documents"

DEPARTMENTS = [
    "Finance", "Operations", "Engineering", "Human Resources",
    "Procurement", "Project Management", "Safety & Security",
    "Legal & Compliance", "Information Technology", "Executive Management"
]

# OCR setup - update path if Tesseract not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- MODELS ----------
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---------- HELPERS ----------
def ensure_dirs():
    Path(OUTPUT_DIR, "originals").mkdir(parents=True, exist_ok=True)
    for dept in DEPARTMENTS:
        Path(OUTPUT_DIR, "routed", dept, "docs").mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_DIR, "routed", dept, "text").mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_DIR, "routed", dept, "summary").mkdir(parents=True, exist_ok=True)

def save_summary_pdf(text, outpath):
    c = canvas.Canvas(outpath, pagesize=A4)
    width, height = A4
    y = height - 50
    for line in text.split("\n"):
        c.drawString(50, y, line)
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

# ---------- PIPELINE ----------
def process_pdf(pdf_path):
    ensure_dirs()

    # Save original unchanged
    fname = Path(pdf_path).name
    original_out = Path(OUTPUT_DIR, "originals", fname)
    Path(pdf_path).replace(original_out)

    # Convert to images
    images = convert_from_path(str(original_out), dpi=300)

    # Extract per-page text
    page_texts = []
    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img, lang="eng+mal")  # add 'mal' for Malayalam if installed
        page_texts.append(text)

    # Classification per page
    pages_by_dept = defaultdict(list)
    dept_texts = defaultdict(str)

    for i, text in enumerate(page_texts, start=1):
        if not text.strip():
            continue
        result = classifier(text, candidate_labels=DEPARTMENTS, multi_label=False)
        top_dept = result["labels"][0]
        pages_by_dept[top_dept].append(i)
        dept_texts[top_dept] += f"\n--- Page {i} ---\n{text}\n"

    # Split PDF into dept copies
    reader = PdfReader(str(original_out))
    for dept, pages in pages_by_dept.items():
        if not pages:
            continue
        writer = PdfWriter()
        for p in pages:
            writer.add_page(reader.pages[p-1])
        out_pdf = Path(OUTPUT_DIR, "routed", dept, "docs", fname)
        with open(out_pdf, "wb") as f:
            writer.write(f)

    # Save text + summary
    for dept, text in dept_texts.items():
        # raw text
        txt_out = Path(OUTPUT_DIR, "routed", dept, "text", f"{Path(fname).stem}.txt")
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(text)

        # summary
        summary = summarizer(text[:3000], max_length=200, min_length=50, do_sample=False)
        summary_text = summary[0]["summary_text"]

        pdf_out = Path(OUTPUT_DIR, "routed", dept, "summary", f"{Path(fname).stem}_summary.pdf")
        save_summary_pdf(summary_text, pdf_out)

    print("Processing complete. Check:", OUTPUT_DIR)

# ---------- MAIN ----------
if __name__ == "__main__":
    process_pdf(INPUT_PDF)
