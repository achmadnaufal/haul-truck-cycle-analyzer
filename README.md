# Haul Truck Cycle Analyzer

Haul truck cycle time analysis, bottleneck identification, and optimization

## Features
- Data ingestion from CSV/Excel input files
- Automated analysis and KPI calculation
- Summary statistics and trend reporting
- Sample data generator for testing and development

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import HaulTruckAnalyzer

analyzer = HaulTruckAnalyzer()
df = analyzer.load_data("data/sample.csv")
result = analyzer.analyze(df)
print(result)
```

## Data Format

Expected CSV columns: `truck_id, cycle_id, date, load_time_min, haul_time_min, dump_time_min, queue_time_min, cycle_total_min`

## Project Structure

```
haul-truck-cycle-analyzer/
├── src/
│   ├── main.py          # Core analysis logic
│   └── data_generator.py # Sample data generator
├── data/                # Data directory (gitignored for real data)
├── examples/            # Usage examples
├── requirements.txt
└── README.md
```

## License

MIT License — free to use, modify, and distribute.
