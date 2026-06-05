import numpy as np
import random
from fd6.shapegen.pytorch_poc import PyTorchSearcher

target = np.zeros((64, 64, 3), dtype=np.uint8)
canvas = np.zeros((64, 64, 3), dtype=np.uint8)
edge = np.ones((64, 64), dtype=np.float32)

searcher = PyTorchSearcher(target, alpha_mask=None, edge_weight=edge)
rng = random.Random(42)
score, shape = searcher.search(canvas, n_random=10, n_mutate=0, max_size_frac=0.5, rng=rng)

print("Score:", score)
print("Shape:", shape)
