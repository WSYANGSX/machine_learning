import time
from tqdm import tqdm

a = range(100)

pbar = tqdm(enumerate(a), total=len(a))
for i, num in pbar:
    time.sleep(0.1)
    pbar.set_description(f"Num: {num}")

pbar.set_description("Num: 100")