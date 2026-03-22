import argparse
import os
import time
from random import randint, seed

from turbo import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple lm-turbo generation benchmark.")
    parser.add_argument("--num-seqs", type=int, default=4, help="Number of concurrent sequences.")
    parser.add_argument("--max-input-len", type=int, default=64, help="Maximum input length per sequence.")
    parser.add_argument("--max-output-len", type=int, default=64, help="Maximum generated length per sequence.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for prompt construction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed(args.seed)
    min_len = min(100, args.max_input_len, args.max_output_len)

    local_candidates = [
        os.path.expanduser("./models/Qwen/Qwen3-0.6B/"),
    ]
    path = next(
        (candidate for candidate in local_candidates if os.path.isdir(candidate)),
        "Qwen/Qwen3-0.6B",
    )
    llm = LLM(path, enforce_eager=False, max_model_len=args.max_model_len)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(min_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams())
    t0 = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t0
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    print(
        f"num_seqs={args.num_seqs}, max_input_len={args.max_input_len}, "
        f"max_output_len={args.max_output_len}"
    )
    print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
