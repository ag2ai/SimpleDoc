# SimpleDoc

**SimpleDoc** is a lightweight yet powerful Retrieval-Augmented Generation (RAG) framework for multi-modal document understanding, with a focus on Document Visual Question Answering (DocVQA). It introduces a dual-cue retrieval mechanism and an iterative reasoning agent that together outperform more complex multi-agent pipelines, using fewer document pages.

![Architecture](MainFigure.png) <!-- Replace with actual image path -->

## Key Contributions

- **Dual-Cue Retrieval**: Combines page-level visual embeddings and LLM-generated summaries to retrieve and rerank relevant pages.
- **Iterative Reasoning**: A single VLM-based agent dynamically updates queries and working memory to iteratively refine answers.
- **Fewer Pages, Better Accuracy**: Achieves up to **70.12%** accuracy on DocVQA benchmarks while reading only ~3.5 pages per query.
- **Simple but Effective**: Outperforms multi-agent systems like MDocAgent and hybrid RAG pipelines like M3DocRAG on 3 out of 4 major benchmarks.

## Project Structure

├── preprocess/ # Offline embedding and summary extraction
├── retriever/ # Dual-cue retrieval logic
├── reasoner/ # Iterative reasoning and memory updates
├── prompts/ # Structured prompts for all stages
├── configs/ # Configuration and evaluation scripts
├── run_simpledoc.py # Main entry point
└── README.md

## Method Overview

SimpleDoc operates in two distinct stages:

### 1. Offline Document Processing
- Extracts **visual embeddings** per page via models like **ColPali/ColQwen**.
- Generates **summaries** per page using LLMs prompted with structured instructions.
- Stores both into a retrievable vector DB.

### 2. Online Multi-Modal QA Loop
- Embeds a user query and retrieves top-*k* candidate pages by embedding similarity.
- Filters and **re-ranks** these using a summary-aware LLM.
- A **reasoning agent** decides whether the current context suffices to answer or if further refinement is needed.
- The process continues iteratively, updating working memory and queries until the answer is found or the query is deemed unanswerable.

## Requirements

- Python 3.9+
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [ChromaDB](https://github.com/chroma-core/chroma)
- ColPali or ColQwen-2.5 embedding model
- Qwen2.5-VL or compatible VLM
- PDF parser (PyMuPDF or pdfminer.six)
- OpenAI-compatible API (for local or cloud LLMs)

## Quickstart

```bash
git clone https://github.com/yourusername/simpledoc.git
cd simpledoc

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python preprocess/embed_and_summarize.py --input_dir ./pdfs

# Answer questions
python run_simpledoc.py --query "What is the main finding of the study?" --doc ./pdfs/sample.pdf
