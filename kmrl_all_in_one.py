#!/usr/bin/env python3
import os
import io
import sys
import argparse
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import logging
from tqdm import tqdm


from PyPDF2 import PdfReader, PdfWriter
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
import pytesseract
from PIL import Image

# NLP / ML
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
from datasets import Dataset
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_INPUT = "C:/Users/moham/Documents/KMRL- AI/Data/AIIB-APD-Project-Document-Kochi-Metro-Rail-Project-Ph-II.pdf"


OUTPUT_DIR = Path("outputs")
ORIGINALS_DIR = OUTPUT_DIR / "originals"
ROUTED_DIR = OUTPUT_DIR / "routed"
DATASET_CSV = Path("kmrl_dataset.csv")
LABELING_CSV = Path("kmrl_dataset_for_labeling.csv")


TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # update for your system or leave if tesseract in PATH
POPPLER_PATH = None  # e.g. r"C:\poppler-xx\Library\bin" or None if poppler in PATH


CHUNK_SIZE = 1000  # chars per chunk for training
CHUNK_OVERLAP = 200


DEFAULT_CLASS_MODEL = "xlm-roberta-base"  # multilingual base
FAST_CLASS_MODEL = "distilbert-base-multilingual-cased"  # faster / lighter
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # smaller summarizer


TRAIN_EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 256

DEPARTMENTS = [
    "Finance", "Operations", "Engineering", "Human Resources",
    "Procurement", "Project Management", "Safety & Security",
    "Legal & Compliance", "Information Technology", "Executive Management", "General"
]

DEPT_KEYWORDS = {
    "Finance": ["invoice", "payment", "payment", "bill", "gst", "tax", "reconciliation", "budget", "cost", "amount", "voucher"],
    "Operations": ["shift", "schedule", "train", "station", "operat", "daily", "roster", "operator", "service", "timetable"],
    "Engineering": ["maintenance", "inspection", "design", "drawing", "blueprint", "infrastructure", "track", "signaling", "electrical", "rolling stock"],
    "Human Resources": ["employee", "onboarding", "resume", "salary", "payroll", "attendance", "hr", "training", "staff"],
    "Procurement": ["tender", "purchase order", "po", "vendor", "procure", "quotation", "bid", "procurement"],
    "Project Management": ["project", "progress", "milestone", "risk", "timeline", "schedule", "implementation", "dpr"],
    "Safety & Security": ["safety", "incident", "audit", "investigation", "security", "emergency", "hazard"],
    "Legal & Compliance": ["agreement", "contract", "legal", "compliance", "opinion", "certificate", "law"],
    "Information Technology": ["erp", "system", "it", "software", "log", "server", "application", "database"],
    "Executive Management": ["board", "policy", "approval", "strategy", "director", "minutes", "decision"],
    "General": []
}


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    ROUTED_DIR.mkdir(parents=True, exist_ok=True)
    for d in DEPARTMENTS:
        (ROUTED_DIR / d / "docs").mkdir(parents=True, exist_ok=True)
        (ROUTED_DIR / d / "text").mkdir(parents=True, exist_ok=True)
        (ROUTED_DIR / d / "summary").mkdir(parents=True, exist_ok=True)

def try_extract_text_pdf_reader(pdf_path):
    
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for p in reader.pages:
        try:
            text = p.extract_text()
        except Exception:
            text = None
        pages_text.append(text or "")
    return pages_text

def ocr_pdf_pages(pdf_path, dpi=300):
    
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not installed. Please install pdf2image.")
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    poppler_arg = POPPLER_PATH if POPPLER_PATH else None
    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_arg)
    texts = []
    for img in tqdm(images, desc="OCR pages"):
        txt = pytesseract.image_to_string(img, lang="eng+mal")  # change languages as needed
        texts.append(txt)
    return texts

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap if (end - overlap) > start else end
    return chunks

def bootstrap_label_for_text(text):
    
    t = text.lower()
    scores = {}
    for dept, keys in DEPT_KEYWORDS.items():
        s = 0
        for k in keys:
            s += t.count(k.lower())
        scores[dept] = s
    # choose best (if all zeros, General)
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]
    if best_score == 0:
        best = "General"
    return best, best_score

def save_summary_pdf(text, outpath, title="Summary"):
    c = canvas.Canvas(str(outpath), pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for line in text.splitlines():
        if y < margin + 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        # wrap long lines rudimentarily
        for i in range(0, len(line), 120):
            c.drawString(margin, y, line[i:i+120])
            y -= 14
    c.save()


# ----------------- PROCESSING & DATASET CREATION ---------------

def process_and_bootstrap(pdf_path, export_for_labeling=True, chunk_size=CHUNK_SIZE):
    
    ensure_output_dirs()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    
    dest_original = ORIGINALS_DIR / pdf_path.name
    if not dest_original.exists():
        shutil.copy(pdf_path, dest_original)
        logger.info(f"Original copied to {dest_original}")
    else:
        logger.info(f"Original already exists at {dest_original}")

    
    logger.info("Trying text extraction with PyPDF2 (fast)...")
    page_texts = try_extract_text_pdf_reader(dest_original)
    total_chars = sum(len(p) for p in page_texts)
    logger.info(f"PyPDF2 extracted total {total_chars} characters from {len(page_texts)} pages")
    
    if total_chars < 500:
        logger.info("Low text count -> falling back to Tesseract OCR (this will be slower).")
        page_texts = ocr_pdf_pages(dest_original, dpi=300)
    rows = []


    for p_idx, page_text in enumerate(page_texts, start=1):
        chunks = chunk_text(page_text, size=chunk_size)
        if not chunks:
            chunks = [page_text]
        for c_idx, chunk in enumerate(chunks, start=1):
            label, score = bootstrap_label_for_text(chunk)
            rows.append({
                "page": p_idx,
                "chunk_id": f"{p_idx}_{c_idx}",
                "text": chunk,
                "label": label,
                "bootstrap_score": score
            })

    
    fieldnames = ["page", "chunk_id", "text", "label", "bootstrap_score"]
    with open(DATASET_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info(f"Bootstrapped dataset written to {DATASET_CSV} ({len(rows)} samples)")

    
    if export_for_labeling:
        
        with open(LABELING_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["page","chunk_id","label","bootstrap_score","text_preview","full_text"])
            w.writeheader()
            for r in rows:
                text_preview = (r["text"][:400] + "...") if len(r["text"])>400 else r["text"]
                w.writerow({
                    "page": r["page"],
                    "chunk_id": r["chunk_id"],
                    "label": r["label"],
                    "bootstrap_score": r["bootstrap_score"],
                    "text_preview": text_preview,
                    "full_text": r["text"]
                })
        logger.info(f"Exported {LABELING_CSV} for manual correction. Edit label column and save as corrected_labels.csv to retrain.")

    return DATASET_CSV


def train_classifier(dataset_csv_path, model_name=DEFAULT_CLASS_MODEL, output_dir="./kmrl_bert_model", epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
    
    ds = Dataset.from_pandas(__load_csv_as_df(dataset_csv_path))
    # use str labels
    labels = sorted(list({r["label"] for r in ds}))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    def map_labels(batch):
        batch["labels"] = [label2id.get(l, label2id["General"]) for l in batch["label"]]
        return batch

    ds = ds.map(map_labels, batched=True)
    # split
    ds = ds.train_test_split(test_size=0.15, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
    output_dir=output_dir,
    do_eval=True,   # old flag for evaluation
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_steps=50
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels_eval, preds)
        f1 = f1_score(labels_eval, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logger.info("Starting training... (this may take time on CPU)")
    trainer.train()
    logger.info("Training complete. Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    # evaluation report
    logger.info("Evaluating on test set...")
    preds_output = trainer.predict(ds["test"])
    logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(logits, axis=-1)
    from sklearn.metrics import classification_report
    from sklearn.utils.multiclass import unique_labels
    unique = unique_labels(y_true, y_pred)
    report = classification_report(
    y_true, y_pred,
    labels=unique,
    target_names=[id2label[i] for i in unique],
    zero_division=0
    )
    print(report)
    logger.info("Classification report:\n" + report)
    return output_dir

def __load_csv_as_df(path):
    import pandas as pd
    df = pd.read_csv(path)
    # if text column is "full_text" from labeling CSV convert it
    if "full_text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"full_text": "text"})
    # ensure text and label exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns (or 'full_text' and 'label').")
    # drop empty text rows
    df = df[~df["text"].isnull()]
    df = df.reset_index(drop=True)
    return df


def classify_and_route(pdf_path, model_dir, summarizer_model=SUMMARIZER_MODEL):
    
    ensure_output_dirs()
    pdf_path = Path(pdf_path)
    dest_original = ORIGINALS_DIR / pdf_path.name
    if not dest_original.exists():
        shutil.copy(pdf_path, dest_original)

    # extract page texts (try PyPDF2 -> OCR)
    page_texts = try_extract_text_pdf_reader(dest_original)
    if sum(len(p) for p in page_texts) < 500:
        page_texts = ocr_pdf_pages(dest_original, dpi=300)

    # load classifier pipeline
    classifier_pipe = pipeline("text-classification", model=model_dir, tokenizer=model_dir, return_all_scores=False)

    # summarizer
    summarizer_pipe = pipeline("summarization", model=summarizer_model)

    pages_by_dept = defaultdict(list)
    dept_texts = defaultdict(str)

    # classify per-page (we'll chunk inside if needed)
    for p_idx, p_text in enumerate(page_texts, start=1):
        if not p_text.strip():
            continue
        # chunk page into pieces for robust classification
        chunks = chunk_text(p_text, size=1500, overlap=200)
        # classify each chunk and aggregate
        chunk_labels = []
        for ch in chunks:
            try:
                out = classifier_pipe(ch[:2000])  # limit length
                # pipeline returns list of dict
                label = out[0]['label'] if isinstance(out, list) else out['label']
            except Exception as e:
                logger.warning("Classification pipeline failed on chunk; fallback to bootstrap")
                label, _ = bootstrap_label_for_text(ch)
            chunk_labels.append(label)
        # majority vote
        if chunk_labels:
            label_counts = Counter(chunk_labels)
            top_label = label_counts.most_common(1)[0][0]
        else:
            top_label = "General"
        pages_by_dept[top_label].append(p_idx)
        dept_texts[top_label] += f"\n--- Page {p_idx} ---\n{p_text}\n"

    # Split original PDF pages into per-dept PDFs
    reader = PdfReader(str(dest_original))
    for dept, pages in pages_by_dept.items():
        if not pages:
            continue
        writer = PdfWriter()
        for p in pages:
            try:
                writer.add_page(reader.pages[p-1])
            except Exception:
                logger.warning(f"Unable to add page {p} to dept {dept}")
        out_pdf_path = ROUTED_DIR / dept / "docs" / dest_original.name
        with open(out_pdf_path, "wb") as f:
            writer.write(f)
        logger.info(f"Wrote {len(pages)} pages to {out_pdf_path}")

    # Save extracted text & summary per dept
    for dept, text in dept_texts.items():
        if not text.strip():
            continue
        txt_path = ROUTED_DIR / dept / "text" / (dest_original.stem + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # summarization (may be slow)
        try:
            summary = summarizer_pipe(text[:12000], max_length=200, min_length=40, do_sample=False)
            summary_text = summary[0]["summary_text"]
        except Exception as e:
            logger.warning(f"Summarizer failed: {e}. Using extractive fallback.")
            # extractive fallback: first 6 lines
            summary_text = "\n".join(text.splitlines()[:12])
        pdf_summary_path = ROUTED_DIR / dept / "summary" / (dest_original.stem + "_summary.pdf")
        save_summary_pdf(summary_text, pdf_summary_path, title=f"Summary - {dest_original.name} - {dept}")
        logger.info(f"Saved summary PDF to {pdf_summary_path}")

    logger.info("Classification & routing complete. Check outputs/ folder.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input PDF", default=DEFAULT_INPUT)
    parser.add_argument("--process", action="store_true", help="Extract text and bootstrap labels (creates kmrl_dataset.csv and kmrl_dataset_for_labeling.csv)")
    parser.add_argument("--train", action="store_true", help="Train classifier on dataset CSV (kmrl_dataset.csv or provided via --labels)")
    parser.add_argument("--labels", help="Path to CSV with corrected labels (use instead of bootstrapped kmrl_dataset.csv)")
    parser.add_argument("--model_dir", help="Where to save / load model", default="./kmrl_bert_model")
    parser.add_argument("--infer", action="store_true", help="Run inference & routing using model_dir")
    parser.add_argument("--all", action="store_true", help="Do process -> train -> infer in one run (slow)")
    args = parser.parse_args()

    if args.process or args.all:
        logger.info("Processing and bootstrapping dataset...")
        process_and_bootstrap(args.input)

    if args.train or args.all:
        labels_csv = args.labels if args.labels else DATASET_CSV
        if not Path(labels_csv).exists():
            logger.error(f"Labels CSV not found: {labels_csv}. Cannot train.")
            sys.exit(1)
        logger.info(f"Training model on {labels_csv} ...")
        train_classifier(labels_csv, model_name=DEFAULT_CLASS_MODEL, output_dir=args.model_dir)

    if args.infer or args.all:
        if not Path(args.model_dir).exists():
            logger.error(f"Model directory {args.model_dir} does not exist. Train or provide correct path.")
            sys.exit(1)
        classify_and_route(args.input, args.model_dir)

if __name__ == "__main__":
    main()
