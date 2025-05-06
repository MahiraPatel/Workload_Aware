# Workload-Aware Multi-Layer Adaptive Bloom Filter

This project implements a **Multi-Layer Adaptive Bloom Filter** in Python designed to improve filtering accuracy and efficiency by adapting to workload characteristics. The filter dynamically promotes items across multiple Bloom filter layers based on observed false positive rates, reducing false positives over time.

## Features

- **SimpleBloomFilter**: A basic Bloom filter implementation using multiple hash functions.
- **MultiLayerAdaptiveBloomFilterFixed**: A multi-layer adaptive Bloom filter with fixed layers and promotion thresholds.
- **Zipfian Workload Generator**: Generates synthetic Zipfian-distributed query data to simulate realistic workloads.
- **Evaluation Framework**: Measures false positives, false negatives, true positives, true negatives, false positive rate, and average query latency.
- **Batch Logging and Visualization**: Logs metrics in batches and plots false positive rates and query latencies over time.
- **Real-Trace-Like Dataset Simulation**: Simulates a workload with "hot" and "cold" domains to mimic real-world query patterns.
- **Parameter Sweep**: Experiments with different numbers of layers and promotion thresholds to analyze performance trade-offs.
- **Static Bloom Filter Comparison**: Compares the adaptive filter against a traditional static Bloom filter.

## Requirements

- Python 3.7+
- numpy
- matplotlib
- pandas

You can install the required packages using pip:

```bash
pip install numpy matplotlib pandas
```

## Usage

The main script `workload_aware_ds.py` runs several experiments:

1. Initializes the multi-layer adaptive Bloom filter.
2. Inserts a set of true positive items.
3. Generates mixed workloads combining true positives and synthetic false positives.
4. Evaluates the filter's accuracy and latency.
5. Logs metrics in batches and plots:
   - False Positive Rate per batch
   - Average Query Latency per batch
6. Simulates a real-trace-like dataset with hot and cold domains.
7. Performs a parameter sweep over different layer counts and promotion thresholds.
8. Compares results with a static Bloom filter.

To run the script first you need to import dataset (https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-urls) 
in your working directory before running the program, and then simply execute:

```bash
python workload_aware_ds.py
```

The script will output evaluation metrics to the console and display plots visualizing the filter's performance.

## Code Overview

- **SimpleBloomFilter**: Implements a Bloom filter with configurable capacity and false positive rate.
- **MultiLayerAdaptiveBloomFilterFixed**: Contains multiple Bloom filter layers with increasing capacity and decreasing false positive rates. Items are promoted to higher layers based on false positive counters and thresholds.
- **generate_zipfian_data**: Utility function to generate Zipfian-distributed keys for synthetic workloads.
- **Evaluation and Logging**: The script evaluates the filter on synthetic and real-trace-like datasets, logging metrics and plotting results.
- **Parameter Sweep**: Tests different configurations to find optimal layer counts and promotion thresholds.

## Results

The script prints detailed statistics including:

- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)
- False Positive Rate (FPR)
- Average Query Latency (in seconds and microseconds)
- Total number of queries processed

It also generates plots showing how the false positive rate and query latency evolve over batches of queries.

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or suggestions, please open an issue or contact the author at mahima02patel@gmail.com
