import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


cache_dir = Path("libero_slot_cache/cache")

index = pq.read_table(cache_dir / "index.parquet")
with open(cache_dir / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
first_row = index.slice(0, 1).to_pylist()[0]
features = np.load(cache_dir / first_row["shard_path"], mmap_mode=None)

print(features.shape, features.dtype, index.num_rows)
