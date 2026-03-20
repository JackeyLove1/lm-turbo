from turbo import LLM, SamplingParams


def main():
    llm = LLM("./models/Qwen/Qwen3-0.6B", enforce_eager=True, max_model_len=512, max_num_seqs=4, max_num_batched_tokens=1024)
    out = llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.6, max_tokens=2), use_tqdm=False)
    print(out)


if __name__ == "__main__":
    main()
