import os
import argparse

import pandas as pd
from loguru import logger
from datasets import load_dataset

from client.client import Lean4Client, batch_verify_proof
from utils.proof_utils import analyze


def format_header(row):
    if not row.header:
        return "import Mathlib\n\n"
    else:
        return row.header.split("theorem")[0].split("/-")[0].strip() + "\n\n"


def formatting(row):
    if "### Complete Lean 4 Proof" in row["DeepSeek-Prover-V2-7B"]:
        text = row["DeepSeek-Prover-V2-7B"].split("### Complete Lean 4 Proof")[-1].strip()
        if text.count("```") == 2:
            return format_header(row) + text.split("```lean4")[-1].split("```")[0].strip()
        else:
            return "lean_cutoff"
    else:
        return "generation_cutoff"


def benchmark_api(client, data_path: str, timeout: int, batch_size: int, num_proc: int):
    """Benchmark the Lean4 verification API by testing it on a dataset of proofs.

    This function loads proof samples from the Goedel-LM/Lean-workbook-proofs dataset,
    verifies them using the Lean4 client, and analyzes the results.

    Args:
        client (Lean4Client): The client instance used to connect to the Lean4 server.
        n (int): Number of samples to test from the dataset.
        timeout (int): Maximum time in seconds allowed for each verification.
        batch_size (int): Number of samples to process in each batch.
        num_proc (int): Number of concurrent processes to use for verification.

    Returns:
        None: Results are printed to stdout by the analyze function.
    """
    dataset = load_dataset(data_path, split="train").to_pandas().iloc[:100]
    dataset["lean4"] = [formatting(row) for _,row in dataset.iterrows()]
    
    samples = [
        {"custom_id": f"question_{i}", "proof": row["lean4"]}
        for i,row in dataset.iterrows()
    ]

    result = batch_verify_proof(
        samples=samples,
        client=client,
        timeout=timeout,
        num_proc=num_proc,
        batch_size=batch_size,
    )

    res_output = pd.DataFrame(result)
    res_output = res_output.sort_values(
        by='custom_id',
        key=lambda x: x.str.split('_').str[-1].astype(int),
        ascending=True
    )

    analyze_output = analyze([dict(row) for _,row in res_output.iterrows()])

    dataset["validation_response"] = list(analyze_output["response"])
    dataset["valid"] = [row["lean4"] if row["lean4"] in ["lean_cutoff", "generation_cutoff"] else analyze_output.loc[i, "valid"] for i,row in dataset.iterrows()]
    dataset.to_json("analyze_result.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Cartinoe5930/DeepSeek-Prover-V2-generation")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:12332")
    args = parser.parse_args()

    num_proc = os.cpu_count()

    logger.info("Evaluation Start!")

    client = Lean4Client(base_url=args.url, disable_cache=False)

    benchmark_api(client, args.data_path, args.timeout, args.batch_size, num_proc)
