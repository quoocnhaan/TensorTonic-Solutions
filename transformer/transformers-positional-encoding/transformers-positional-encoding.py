import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    freq_size = d_model / 2
    freq = np.arange(0, freq_size, 1)

    freq = [(1/(10000**((2*i)/d_model))) for i in freq]
    freq = np.repeat(freq, 2)


    positions = np.arange(0, seq_length, 1)
    encoded_pos = []

    for pos in positions:
        tmp = pos * freq
        tmp[1::2] = np.cos(tmp[1::2])
        tmp[0::2] = np.sin(tmp[0::2])
        encoded_pos.append(tmp)
    
    return np.array(encoded_pos)