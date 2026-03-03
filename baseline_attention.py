import math

import torch
import torch.nn.functional as F


BASELINE_SOURCE_URLS = """```text
https://nlp.seas.harvard.edu/2018/04/03/attention.html
https://gist.github.com/Kaixhin/dc6f73099334a5d41d20804e70ae7f7b
```"""


def attention_baseline(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Baseline scaled dot-product attention.
    # Inputs: Q,K,V shape [L,d], float32, contiguous.
    # Output: O shape [L,d]
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    P = F.softmax(scores, dim=-1)
    return torch.matmul(P, V)
