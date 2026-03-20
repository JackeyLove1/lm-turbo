import torch

from turbo.config import HiddenAct, ModelConfig, RotaryConfig
from turbo.qwen3 import Qwen3ForCausalLM


def _test_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=8,
        hidden_size=32,
        intermediate_size=64,
        rotary_config=RotaryConfig(head_dim=8, rotary_dim=8, max_position=128),
        hidden_act=HiddenAct.SILU,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_token=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3",
        vocab_size=128,
        torch_dtype=torch.float32,
        use_qk_norm=False,
    )


def test_kvcache_matches_full_forward_logits():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_test_config()).eval()

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    next_token = torch.tensor([[5]], dtype=torch.long)
    full_input = torch.cat([prompt, next_token], dim=-1)

    with torch.no_grad():
        full_logits, _ = model(
            full_input,
            position_ids=model.build_position_ids(full_input),
            use_cache=False,
        )
        _, past_kv_cache = model(
            prompt,
            position_ids=model.build_position_ids(prompt),
            use_cache=True,
        )
        cached_logits, updated_cache = model(
            next_token,
            position_ids=model.build_position_ids(next_token, past_kv_cache),
            past_kv_cache=past_kv_cache,
            use_cache=True,
        )

    assert updated_cache is not None
    assert all(layer_cache is not None for layer_cache in updated_cache)
    assert torch.allclose(full_logits[:, -1, :], cached_logits[:, -1, :], atol=1e-5, rtol=1e-4)


def test_kvcache_matches_full_forward_across_multiple_decode_steps():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(_test_config()).eval()

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    continuation = torch.tensor([[5, 6, 7]], dtype=torch.long)
    full_input = torch.cat([prompt, continuation], dim=-1)

    with torch.no_grad():
        full_logits, _ = model(
            full_input,
            position_ids=model.build_position_ids(full_input),
            use_cache=False,
        )
        _, past_kv_cache = model(
            prompt,
            position_ids=model.build_position_ids(prompt),
            use_cache=True,
        )

        step_logits = []
        for token in continuation[0]:
            current = token.view(1, 1)
            cached_logits, past_kv_cache = model(
                current,
                position_ids=model.build_position_ids(current, past_kv_cache),
                past_kv_cache=past_kv_cache,
                use_cache=True,
            )
            step_logits.append(cached_logits[:, -1, :])

    stacked_step_logits = torch.stack(step_logits, dim=1)
    assert past_kv_cache is not None
    assert torch.allclose(full_logits[:, -3:, :], stacked_step_logits, atol=1e-5, rtol=1e-4)
