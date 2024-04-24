"""Benchmark online serving throughput.
"""
import argparse
import asyncio
import json
import random
import time
import histoprint
from typing import AsyncGenerator, List, Tuple, Optional
import os
import pandas as pd
import _pickle as pickle

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from lib.structs import TestRequest, Dataset, RequestResult

pbar: Optional[tqdm] = None

def sample_requests(dataset_path: str, num_prompts: int) -> List[TestRequest]:
    """
    sample_requests: Sample the given number of requests from the dataset.
    """
    dataset = Dataset.load(dataset_path)
    if num_prompts > len(dataset.data):
        raise ValueError(
            f"Number of prompts ({num_prompts}) is larger than the dataset size ({len(dataset.data)})."
        )
    return random.sample(dataset.data, num_prompts)


async def get_request(
    input_requests: List[TestRequest],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
) -> AsyncGenerator[TestRequest, None]:
    interval_lens = len(input_requests)
    input_requests = iter(input_requests)

    if request_rate not in [float("inf"), 0.0]:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        else:
            raise ValueError(
                f"Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
    for idx, request in enumerate(input_requests):
        yield request
        if request_rate == float("inf") or request_rate == 0.0:
            continue

        interval = intervals[idx]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> RequestResult:
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "disstserve" or backend == "vllm" or backend == "fastertransformer":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # The maximum length of the input is 2048, limited by the embedding
    # table size.
    assert prompt_len+output_len < 2048
    
    request_start_time = time.time()
    request_output = None

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            try:
                output = json.loads(output)
            except:
                print("Failed to parse the response:")
                print(output)
                continue

            # Re-send the request if it failed.
            if "error" not in output:
                request_output = output
                break
            else:
                print(f"Failed to process the request: {output['error']}")
                print(f"Resending the request: {pload}")

    request_end_time = time.time()
    
    global pbar
    pbar.update(1)
    
    return RequestResult(
        prompt_len,
        output_len,
        request_start_time,
        request_end_time,
        token_timestamps=request_output["timestamps"],
        lifetime_events=request_output.get("lifetime_events", None)
    )


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[TestRequest],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
) -> List[RequestResult]:
    tasks: List[asyncio.Task] = []
    async for request in get_request(
        input_requests, process_name, request_rate, request_cv
    ):
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                request.prompt,
                request.prompt_len,
                request.output_len,
                best_of,
                use_beam_search,
            )
        )
        tasks.append(task)
    request_results = await asyncio.gather(*tasks)
    return request_results


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    input_requests = sample_requests(
        args.dataset, args.num_prompts
    )
    print("Sampling done. Start benchmarking...")

    global pbar
    pbar = tqdm(total=args.num_prompts)
    benchmark_start_time = time.time()
    request_results = asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            args.request_cv,
            args.process_name,
        )
    )
    benchmark_end_time = time.time()
    pbar.close()
    
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput:")
    print(f"\t{args.num_prompts / benchmark_time:.2f} requests/s")
    print(f"\t{sum([req.prompt_len + req.output_len for req in input_requests]) / benchmark_time:.2f} tokens/s")
    print(f"\t{sum([req.output_len for req in input_requests]) / benchmark_time:.2f} output tokens/s")
    
    agreements = [0.85, 0.9, 0.95, 0.98]
    print("Distribution of first token latency:")
    first_token_latencies = [req.ftl for req in request_results]
    histoprint.print_hist(
        np.histogram(
            np.array(first_token_latencies),
            bins = 20,
            range = (min(first_token_latencies), max(first_token_latencies))
        ),
        bg_colors="c",
    )
    for agreement in agreements:
        print(f"\t{agreement*100:.0f}%: {np.quantile(first_token_latencies, agreement)} s") 
    print()

    print("Distribution of TPOT:")
    decoding_tpots = [req.tpot for req in request_results]
    histoprint.print_hist(
        np.histogram(
            np.array(decoding_tpots),
            bins = 20,
            range = (min(decoding_tpots), max(decoding_tpots))
        ),
        bg_colors="c",
    )
    for agreement in agreements:
        print(f"\t{agreement*100:.0f}%: {np.quantile(decoding_tpots, agreement)} s")

    with open(args.output, "w") as f:
        json.dump(request_results, f, default=vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend", type=str, default="distserve", choices=["distserve", "vllm", "tgi", "fastertransformer"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the (preprocessed) dataset."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--request-cv",
        type=float,
        default=1.0,
        help="the coefficient of variation of the gap between" "the requests.",
    )
    parser.add_argument(
        "--process-name",
        type=str,
        default="possion",
        choices=["possion", "gamma", "uniform"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="The path to the output file that stores the benchmark results."
    )
    
    args = parser.parse_args()
    
    main(args)