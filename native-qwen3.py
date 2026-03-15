"""
Qwen3 Dense 模型从零实现（PyTorch），并加载 HuggingFace 权重运行推理
支持: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B 等 Dense 变体
"""

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 1. 模型配置
# ─────────────────────────────────────────────

@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8        # GQA: KV head 数量
    head_dim: int = 64                  # 每个 head 的维度
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True

    # Qwen3 特有: QK Norm
    use_qk_norm: bool = True

    @classmethod
    def from_pretrained_config(cls, config_dict: dict) -> "Qwen3Config":
        """从 HuggingFace config.json 加载配置"""
        return cls(
            vocab_size=config_dict.get("vocab_size", 151936),
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict.get("num_key_value_heads", config_dict["num_attention_heads"]),
            head_dim=config_dict.get("head_dim", config_dict["hidden_size"] // config_dict["num_attention_heads"]),
            max_position_embeddings=config_dict.get("max_position_embeddings", 40960),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            rope_theta=config_dict.get("rope_theta", 1000000.0),
            attention_bias=config_dict.get("attention_bias", False),
            mlp_bias=config_dict.get("mlp_bias", False),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
        )


# ─────────────────────────────────────────────
# 2. 基础模块
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (x_normed * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码"""

    def __init__(self, dim: int, max_seq_len: int = 40960, theta: float = 1000000.0):
        super().__init__()
        # 计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将 x 的后半部分取负并与前半部分拼接"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 应用 RoPE"""
    # cos/sin: [seq_len, head_dim]  →  [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    if position_ids is not None:
        cos = cos[:, :, position_ids, :]
        sin = sin[:, :, position_ids, :]

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ─────────────────────────────────────────────
# 3. GQA 注意力模块
# ─────────────────────────────────────────────

class Qwen3Attention(nn.Module):
    """
    Grouped Query Attention (GQA) + QK Norm
    Qwen3 相对 LLaMA 的关键区别：对 Q/K 进行 RMSNorm
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # GQA 分组倍数

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Qwen3 特有：Q/K 归一化
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # ① 线性投影
        q = self.q_proj(hidden_states)  # [B, S, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]

        # ② reshape: [B, S, H, D] → [B, H, S, D]
        q = q.view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ③ QK Norm（Qwen3 特有）
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ④ RoPE
        cos, sin = self.rotary_emb(S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # ⑤ GQA：将 KV 扩展至与 Q 一致的 head 数量
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # ⑥ Scaled dot-product attention（使用 PyTorch 2.0 flash attention）
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )

        # ⑦ 合并 heads 并输出投影
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ─────────────────────────────────────────────
# 4. MLP (SwiGLU)
# ─────────────────────────────────────────────

class Qwen3MLP(nn.Module):
    """SwiGLU FFN：out = SiLU(gate_proj(x)) * up_proj(x)，再 down_proj"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────
# 5. Transformer 层
# ─────────────────────────────────────────────

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm + 残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ─────────────────────────────────────────────
# 6. 完整模型
# ─────────────────────────────────────────────

class Qwen3Model(nn.Module):
    """Qwen3 Transformer 主体（不含 LM head）"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 因果语言模型（含 LM head）"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定：embedding 与 lm_head 共享参数
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, attention_mask, position_ids)
        return self.lm_head(hidden)   # [B, S, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 151645,
    ) -> torch.Tensor:
        """简单的自回归生成（贪心 + top-p）"""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)          # [B, S, V]
            next_logits = logits[:, -1, :]            # [B, V]

            # temperature scaling
            next_logits = next_logits / temperature

            # top-p (nucleus) sampling
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            mask = (cumulative - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_ids.gather(-1, torch.multinomial(sorted_probs, 1))

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated


# ─────────────────────────────────────────────
# 7. 加载 HuggingFace 权重
# ─────────────────────────────────────────────

def remap_hf_key(hf_key: str) -> str:
    """
    将 HuggingFace state_dict 的 key 映射到我们自定义模型的 key
    HF:  model.layers.0.self_attn.q_proj.weight
    Ours: model.layers.0.self_attn.q_proj.weight  (本实现命名与 HF 一致，无需重映射)
    """
    return hf_key   # 本实现命名已与 HuggingFace 对齐


def load_qwen3_from_hf(model_name_or_path: str, device: str = "cpu") -> Tuple[Qwen3ForCausalLM, object]:
    """
    从 HuggingFace Hub 或本地路径加载 Qwen3 模型权重

    用法:
        model, tokenizer = load_qwen3_from_hf("Qwen/Qwen3-0.6B")
    """
    import os

    from transformers import AutoTokenizer

    # ── 读取 config ──
    if os.path.isdir(model_name_or_path):
        config_path = os.path.join(model_name_or_path, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_name_or_path, "config.json")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = Qwen3Config.from_pretrained_config(config_dict)
    print(f"[Config] layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, hidden={config.hidden_size}")

    # ── 初始化模型 ──
    model = Qwen3ForCausalLM(config)

    # ── 加载 safetensors 或 .bin 权重 ──
    try:
        import glob

        from safetensors.torch import load_file

        if os.path.isdir(model_name_or_path):
            weight_files = sorted(glob.glob(os.path.join(model_name_or_path, "*.safetensors")))
        else:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(model_name_or_path)
            weight_files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))

        state_dict = {}
        for wf in weight_files:
            print(f"  Loading {wf} ...")
            state_dict.update(load_file(wf, device=device))

    except ImportError:
        print("safetensors 未安装，尝试加载 pytorch_model.bin ...")
        if os.path.isdir(model_name_or_path):
            bin_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        else:
            from huggingface_hub import hf_hub_download
            bin_path = hf_hub_download(model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(bin_path, map_location=device)

    # ── 加载权重（允许部分不匹配，用于调试）──
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print("[OK] 权重加载完成")

    # ── 加载 tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path if os.path.isdir(model_name_or_path) else model_name_or_path,
        trust_remote_code=False
    )

    model.eval()
    model.to(device)
    return model, tokenizer


# ─────────────────────────────────────────────
# 8. 推理入口
# ─────────────────────────────────────────────

def chat(model: Qwen3ForCausalLM, tokenizer, prompt: str, device: str = "cpu"):
    """单轮对话（非 thinking 模式）"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=False  # 若只想要非思考模式取消注释
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    print(f"Input tokens: {input_ids.shape[1]}")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


# ─────────────────────────────────────────────
# 9. 主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # 用法: python qwen3.py [model_name_or_path] [device]
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/Qwen/Qwen3-0.6B"
    device     = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_path} on {device} ...")
    model, tokenizer = load_qwen3_from_hf(model_path, device=device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # 推理
    prompt = "用 Python 写一个快速排序算法"
    print(f"\nPrompt: {prompt}")
    print("=" * 50)
    response = chat(model, tokenizer, prompt, device)
    print(response)
