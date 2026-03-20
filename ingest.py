from pathlib import Path
from unstructured.partition.pdf import partition_pdf


# PDF-to-text extractor
data_path = Path(__file__).parent/"data"

all_documents = []

for pdf_file in data_path.glob("*.pdf"):
    elements = partition_pdf(filename=str(pdf_file), strategy="auto", languages=["eng"])
    
    for el in elements:
        if el.text:  
            all_documents.append({
                "source": pdf_file.name,
                "text": el.text,
                "type": el.category  
            })

    # print(f"{pdf_file.name}: {len(elements)} elements")

