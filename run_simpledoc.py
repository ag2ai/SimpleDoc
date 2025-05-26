#!/usr/bin/env python3
"""
AG²-style main pipeline runner for SimpleDoc.
Loads samples, instantiates the SimpleDocAgent, and executes inference.
"""

import os
import json
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from agent.simpledoc_agent import SimpleDocAgent
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from utils.openai_helper import load_args_from_yaml  # utility to replace argparse

def run_pipeline(args):

    # Load samples
    with open(args.input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Optional embedding model setup
    device = None
    processor_emb = model_emb = None
    if args.use_embedding_based_retrieval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_emb = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        model_emb = ColQwen2_5.from_pretrained("vidore/colqwen2.5-v0.2", torch_dtype=torch.bfloat16).to(device).eval()

    # Create agent
    agent = SimpleDocAgent()

    # Run inference
    if args.n_jobs == 1:
        results = [agent.process_sample(s, args, device, processor_emb, model_emb) for s in tqdm(samples)]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(agent.process_sample)(s, args, device, processor_emb, model_emb) for s in tqdm(samples)
        )

    # Save output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Done] Saved results to {args.output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ag2_config.yaml")
    cli_args = parser.parse_args()
    args = load_args_from_yaml(cli_args.config)
    run_pipeline(args)