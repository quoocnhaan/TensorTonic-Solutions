import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    weights = F.softmax(scores / math.sqrt(d), dim=-1)

    output = weights @ V
    return output
