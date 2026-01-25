"""
Example usage of haul-truck-cycle-analyzer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import HaulTruckAnalyzer
from src.data_generator import generate_sample

# Generate sample data
df = generate_sample(200)
print(f"Sample data: {df.shape[0]} rows, {df.shape[1]} columns")

# Run analysis
analyzer = HaulTruckAnalyzer()
result = analyzer.analyze(df)

print("\n=== Analysis Results ===")
print(f"Total records: {result['total_records']}")
if "totals" in result:
    print("Totals:")
    for k, v in list(result["totals"].items())[:5]:
        print(f"  {k}: {v:,.2f}")
if "means" in result:
    print("Means:")
    for k, v in list(result["means"].items())[:5]:
        print(f"  {k}: {v:.3f}")
