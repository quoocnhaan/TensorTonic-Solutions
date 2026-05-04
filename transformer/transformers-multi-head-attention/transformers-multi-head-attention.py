import numpy as np


# -----------------------
# Softmax
# -----------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability trick
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# -----------------------
# Scaled dot-product attention
# -----------------------
def attention(Q, K, V, d_k):
    scores = Q @ K.transpose(0, 1, 3, 2)   # (batch, heads, seq, seq)
    scores = scores / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V  # (batch, heads, seq, d_k)


# -----------------------
# Multi-Head Attention
# -----------------------
def multi_head_attention(Q, K, V,
                         W_q, W_k, W_v,
                         W_o,
                         num_heads):

    batch, seq_len, d_model = Q.shape

    d_k = d_model // num_heads

    # 1. Linear projections
    Q = Q @ W_q   # (batch, seq, d_model)
    K = K @ W_k
    V = V @ W_v

    # 2. Split into heads
    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        # (batch, heads, seq, d_k)

    Q = split_heads(Q)
    K = split_heads(K)
    V = split_heads(V)

    # 3. Attention per head
    heads = attention(Q, K, V, d_k)
    # (batch, heads, seq, d_k)

    # 4. Concatenate heads
    concat = heads.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    # 5. Final linear projection
    output = concat @ W_o

    return output