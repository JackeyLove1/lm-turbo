from jaxtyping import Float
from torch import Tensor

Tensor2D = Float[Tensor, "batch seq"]
Tensor3D = Float[Tensor, "batch seq dim"]
Tensor4D = Float[Tensor, "batch seq head dim"]
AttentionTensor = Float[Tensor, "batch n_heads seq head_dim"]
