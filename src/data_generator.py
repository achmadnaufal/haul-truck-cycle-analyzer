"""
Generate sample data for haul-truck-cycle-analyzer.
Run this script to create test data: python src/data_generator.py
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path


COLUMNS = ["truck_id", "cycle_id", "date", "load_time_min", "haul_time_min", "dump_time_min", "queue_time_min", "cycle_total_min"]

def generate_sample(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic sample dataset."""
    np.random.seed(seed)
    random.seed(seed)
    
    base_date = datetime(2023, 1, 1)
    n_groups = max(5, n // 20)
    
    data = {}
    for i, col in enumerate(COLUMNS):
        if "date" in col:
            data[col] = [
                (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
                for _ in range(n)
            ]
        elif "id" in col or "code" in col:
            data[col] = [f"{col[:3].upper()}{random.randint(1, n_groups):03d}" for _ in range(n)]
        elif "category" in col or "type" in col or "status" in col:
            data[col] = [random.choice(["A", "B", "C", "D"]) for _ in range(n)]
        elif "pct" in col or "rate" in col or "ratio" in col:
            data[col] = np.round(np.random.uniform(0, 100, n), 2)
        else:
            # Positive numeric with realistic distribution
            base = np.random.exponential(100, n)
            noise = np.random.normal(0, 10, n)
            data[col] = np.round(np.abs(base + noise), 2)
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_sample(300)
    out_path = "data/sample.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} records → {out_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
