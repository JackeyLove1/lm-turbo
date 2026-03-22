import time
from random import randint, seed
import sys

from turbo import LLM, SamplingParams


def run_case(max_len: int, num_seqs: int) -> None:
    path = "./models/Qwen/Qwen3-0.6B/"
    print(f"=== test max_len={max_len} num_seqs={num_seqs} ===", flush=True)
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    min_len = min(100, max_len)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(min_len, max_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(min_len, max_len),
        )
        for _ in range(num_seqs)
    ]
    try:
        llm.generate(["Benchmark: "], SamplingParams())
        t0 = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        elapsed = time.time() - t0
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / elapsed
        print(
            f"OK max_len={max_len} num_seqs={num_seqs} total_tokens={total_tokens} "
            f"time={elapsed:.2f}s throughput={throughput:.2f}tok/s",
            flush=True,
        )
    except Exception as exc:
        print(
            f"FAIL max_len={max_len} num_seqs={num_seqs}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        try:
            llm.exit()
        except Exception as exc:
            print(
                f"EXIT_FAIL max_len={max_len} num_seqs={num_seqs}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def main() -> None:
    seed(0)
    if len(sys.argv) > 2:
        max_lens = (int(sys.argv[1]),)
        num_seqs_list = (int(sys.argv[2]),)
    else:
        max_lens = (256, 128, 64)
        num_seqs_list = (16, 8, 4)
    for max_len in max_lens:
        for num_seqs in num_seqs_list:
            run_case(max_len, num_seqs)


if __name__ == "__main__":
    main()
