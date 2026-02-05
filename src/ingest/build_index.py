from pathlib import Path
import os

PDF_DIR = Path(os.getenv("PDF_DIR", "data/raw_pdfs"))

def main():
    pdfs = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {PDF_DIR.resolve()}")

    for pdf in pdfs:
        print(f"Indexing: {pdf.name}")
        # TODO: extract -> chunk -> embed -> upsert to vector DB

    print("Done.")

if __name__ == "__main__":
    main()