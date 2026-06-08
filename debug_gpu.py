import numpy as np
import random
from fd6.shapegen.pytorch_backend import PyTorchDiffRenderer

print("Starting GPU diagnostic...")
try:
    target = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    alpha = np.ones((256, 256), dtype=np.uint8) * 255
    edge = np.ones((256, 256), dtype=np.float32)
    edge_dir = np.zeros((256, 256), dtype=np.float32)

    print("Initializing backend...")
    gpu = PyTorchDiffRenderer(target, alpha, edge, edge_dir)
    gpu._current_types = ["rotated_ellipse", "rotated_rectangle"]

    canvas = np.full((256, 256, 3), 128, dtype=np.uint8)
    rng = random.Random(42)

    print("Running search...")
    score, shape = gpu.search(canvas, n_random=1000, n_mutate=20, max_size_frac=1.0, rng=rng)
    print(f"Success! Score: {score}, Shape: {type(shape).__name__}")

except Exception as e:
    import traceback
    print("CRASH DETECTED!")
    traceback.print_exc()
