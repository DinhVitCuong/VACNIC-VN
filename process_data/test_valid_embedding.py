# CODE FOR TESTING IF THESE EMBEDDING IS VALID

import os
import glob
import random
import numpy as np
from typing import List

def pick_random_files(folder: str, min_n: int = 5, max_n: int = 10) -> List[str]:
    files = glob.glob(os.path.join(folder, "*.npy"))
    if not files:
        print(f"[!] No .npy files found in: {folder}")
        return []
    k = min(len(files), random.randint(min_n, max_n))
    return random.sample(files, k) if len(files) >= k else files

def check_array(arr: np.ndarray, sample_cap: int = 100_000) -> dict:
    info = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "all_finite_sample": True,
        "nan_count_sample": 0,
        "inf_count_sample": 0,
        "min": None,
        "max": None,
    }

    # Basic stats (safe for numeric dtypes)
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        try:
            info["min"] = float(np.nanmin(arr))
            info["max"] = float(np.nanmax(arr))
        except Exception:
            pass

        # Finite check on a sample to avoid huge memory/time
        flat = arr.ravel()
        sample_n = min(sample_cap, arr.size)
        if sample_n > 0:
            idx = np.random.default_rng().choice(arr.size, sample_n, replace=False)
            sample = flat[idx]

            nan_mask = np.isnan(sample) if np.issubdtype(arr.dtype, np.floating) else np.zeros(0, dtype=bool)
            inf_mask = np.isinf(sample) if np.issubdtype(arr.dtype, np.floating) else np.zeros(0, dtype=bool)

            info["nan_count_sample"] = int(nan_mask.sum()) if nan_mask.size else 0
            info["inf_count_sample"] = int(inf_mask.sum()) if inf_mask.size else 0

            if np.issubdtype(arr.dtype, np.floating):
                info["all_finite_sample"] = not (info["nan_count_sample"] or info["inf_count_sample"])

    return info

def validate_npy_folder(folder: str):
    files = pick_random_files(folder, 5, 10)
    if not files:
        return

    ok, fail = 0, 0
    print(f"Checking {len(files)} random .npy files from: {folder}\n")

    for path in files:
        print(f"==> {path}")
        try:
            # mmap_mode='r' avoids loading the entire array into RAM at once
            arr = np.load(path, mmap_mode='r')
            if not isinstance(arr, np.ndarray):
                raise ValueError("Loaded object is not a numpy array")

            info = check_array(arr)

            # Basic validations
            issues = []
            if info["size"] == 0:
                issues.append("empty array")
            if np.issubdtype(arr.dtype, np.floating) and not info["all_finite_sample"]:
                issues.append(f"NaN/Inf found in sample (nan={info['nan_count_sample']}, inf={info['inf_count_sample']})")

            if issues:
                print(f"  [BAD] {', '.join(issues)}")
                print(f"  shape={info['shape']}, dtype={info['dtype']}, min={info['min']}, max={info['max']}\n")
                fail += 1
            else:
                print(f"  [OK ] shape={info['shape']}, dtype={info['dtype']}, size={info['size']}, min={info['min']}, max={info['max']}\n")
                ok += 1

        except Exception as e:
            print(f"  [ERR] Failed to load: {e}\n")
            fail += 1

    print(f"Summary: OK={ok}, FAIL/ERR={fail}")

if __name__ == "__main__":
    # Example: Windows raw string path
    try:
        folder = r"Z:\DATN\data\vacnic_data\embedding\faces"   # <-- change to your folder
        validate_npy_folder(folder)
    except Exception as e:
        print(f"  [ERR] Failed to load folder {folder}: {e}\n")

    try:
        folder = r"Z:\DATN\data\vacnic_data\embedding\objects"   # <-- change to your folder
        validate_npy_folder(folder)
    except Exception as e:
        print(f"  [ERR] Failed to load folder {folder}: {e}\n")
