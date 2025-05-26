#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import fitz  # PyMuPDF
from PIL import Image
import io
from tqdm import tqdm
from glob import glob
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from accelerate import Accelerator

processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")

class PDFPageDataset(Dataset):
    """Dataset for processing PDF pages into images"""
    
    def __init__(self, pdf_paths, image_dpi=150):
        self.pdf_paths = pdf_paths
        self.image_dpi = image_dpi
        
        # Build index of PDF pages
        self.page_index = {}
        self.id_2_doc = []
        for pdf_path in tqdm(pdf_paths, desc="Indexing PDFs"):
            self.page_index[os.path.basename(pdf_path)] = []
            self.id_2_doc.append(os.path.basename(pdf_path))
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                self.page_index[os.path.basename(pdf_path)].append({
                    "pdf_path": pdf_path,
                    "page_num": page_num,
                    "doc_id": os.path.basename(pdf_path)
                })
            doc.close()


        
        print(f"Indexed {len(self.page_index)} pages from {len(pdf_paths)} PDFs")
        
    def __len__(self):
        return len(self.page_index)
    
    def __getitem__(self, idx):
        page_info_list = self.page_index[self.id_2_doc[idx]]
        pdf_path = page_info_list[0]["pdf_path"]
        doc_id = page_info_list[0]["doc_id"]
        doc = fitz.open(pdf_path)
        page_imgs = []
        for page_info in page_info_list:
            # Convert page to image using convert_pdf_pages_to_base64_images logic
            try:
                page = doc.load_page(page_info["page_num"])
                pix = page.get_pixmap(dpi=self.image_dpi)
                page_imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))))
            except Exception as e:
                print(f"Error processing page {page_info['page_num']} of {pdf_path}: {e}, using blank image")
                page_imgs.append(Image.new("RGB", (32, 32), color="white"))

        doc.close()

            # Return the original image separately from other metadata
        return {
            "pdf_path": pdf_path,
            "page_num": [page_info["page_num"] for page_info in page_info_list],  # Convert to 1-indexed
            "doc_id": doc_id,
            "image": page_imgs  # We'll handle this specially in the collate function
        }


# Custom collate function to handle PIL images
def collate_fn(batch):
    batch["image"] = processor.process_images(batch["image"])
    return batch



def process_batch(model, batch):
    """Process a batch of PDF pages and save embeddings"""
    embedding_path = batch["pdf_path"].replace("documents", "embeddings").replace('.pdf', '.pt')
    if os.path.exists(embedding_path):
        print(f"Embedding already exists for {batch['pdf_path']}, skipping.")
        return

    with torch.no_grad():
        batch_images = batch["image"].to(model.device)
        image_embeddings = model(**batch_images)


    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    torch.save(image_embeddings, embedding_path)


def main():
    parser = argparse.ArgumentParser(description="Process PDF documents into embeddings using accelerate")
    parser.add_argument("--input_dir", type=str, default="data/MMLongBench", help="Input directory containing PDF documents")
    parser.add_argument("--image_dpi", type=int, default=150, help="DPI for rendering PDF pages")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()

    
    # Get PDF files
    pdf_files = sorted(glob(f"{args.input_dir}/*/*.pdf", recursive=True))
    
    # Create dataset and dataloader
    print(f"Processing {len(pdf_files)} PDF files")
    
    # Create dataset and dataloader
    dataset = PDFPageDataset(pdf_files, image_dpi=args.image_dpi)
    dataloader = DataLoader(
        dataset, 
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    # Process batches
    for batch in tqdm(dataloader, desc=f"Processing PDF Documents"):
        process_batch(model, batch)

if __name__ == "__main__":
    main()