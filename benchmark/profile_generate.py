from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.profiler as profiler

from turbo import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile lm-turbo generation with torch.profiler.")
    parser.add_argument(
        "--model",
        default="./models/Qwen/Qwen3-0.6B/",
        help="Model path passed to turbo.LLM.",
    )
    parser.add_argument(
        "--trace-path",
        default="trace.json",
        help="Chrome trace output path.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Model dtype, for example float16 or bfloat16.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--warmup-prompt",
        default="Benchmark: ",
        help="Warmup prompt used before the profiled generate call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum output tokens per request.",
    )
    parser.add_argument(
        "--repeat-prompts",
        type=int,
        default=1,
        help="Repeat the prompt set this many times.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt text. Can be provided multiple times.",
    )
    return parser.parse_args()


def default_prompts() -> list[str]:
    return [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
        "Please introduce lm-turbo framework",
    ]


def make_profiler() -> profiler.profile:
    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(profiler.ProfilerActivity.CUDA)
    return profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    prompts = args.prompts or default_prompts()
    prompts = prompts * max(args.repeat_prompts, 1)
    trace_path = Path(args.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    with make_profiler() as prof:
        llm = LLM(
            args.model,
            dtype=args.dtype,
            enforce_eager=False,
            max_model_len=args.max_model_len,
        )
        llm.generate([args.warmup_prompt], SamplingParams(max_tokens=8))
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    prof.export_chrome_trace(str(trace_path))
    print(f"Saved trace to {trace_path.resolve()}")
    print(f"Generated {len(outputs)} responses.")


if __name__ == "__main__":
    main()
