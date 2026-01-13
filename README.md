# Multi-Class Storage Classification Framework

A comprehensive machine learning framework for optimizing cloud storage tier classification using multi-class prediction models. This project implements various classification algorithms to predict optimal storage tiers (COLD, WARM, HOT) based on access patterns, volumes, and cost-latency trade-offs.

## Overview

This framework addresses the challenge of optimizing cloud storage costs and performance by automatically classifying data objects into appropriate storage tiers. It uses historical access patterns and volume data to train machine learning models that can predict the optimal storage tier for each object, balancing cost and latency requirements.

## Features

- **Multi-class Classification**: Predicts three storage tiers (COLD, WARM, HOT)
- **Multiple ML Models**: Supports various algorithms including SVM, Random Forest, KNN, Decision Trees, and more
- **Cost-Latency Optimization**: Configurable cost-weight parameter to balance between cost and performance
- **Time Window Analysis**: Sliding window approach for temporal data analysis
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, cost, and latency
- **Baseline Strategies**: Includes Always Hot, Always Warm, Always Cold, and Online strategies for comparison
- **Sensitivity Analysis**: Tools for analyzing model performance across different parameters

## Project Structure

```
multi-class/
├── main.py                 # Main execution script
├── config/
│   ├── config.yaml        # Configuration file
│   └── config_example.yaml # Example configuration
├── data/                  # Data files directory
│   ├── access_Pop*.txt    # Access pattern data
│   ├── vol_bytes_Pop*.txt # Volume data
│   ├── target_Pop*.txt    # Target labels
│   └── nsJanelas_Pop*.txt # Time window data
├── models/
│   ├── classifiers.py     # ML model implementations
│   ├── user_profiles.py   # User profile and cost models
│   └── onl.py            # Online prediction logic
├── analysis/              # Analysis and visualization tools
│   ├── sensitivity_analysis.py
│   ├── plot_trends.py
│   └── pareto_plot.py
├── results/               # Output results
└── README_cost_weight.md  # Cost weight documentation
```

## Installation

### Prerequisites

- Python 3.7+
- Required packages (install via pip):

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-class
```

2. Ensure your data files are in the `data/` directory with the correct naming convention:
   - `access_Pop{1,2,3,4}.txt` - Access pattern data
   - `vol_bytes_Pop{1,2,3,4}.txt` - Volume data
   - `target_Pop{1,2,3,4}.txt` - Target labels
   - `nsJanelas_Pop{1,2,3,4}.txt` - Time window data

## Configuration

Edit `config/config.yaml` to configure your experiment:

```yaml
# Window configuration
window_size: 4          # Number of weeks per window
step_size: 4            # Step size between windows

# Population to analyze
pop_name: 'Pop2'        # Population name (Pop1, Pop2, Pop3, Pop4)

# Models to run
models_to_run:
  - 'AHL'    # Always Hot
  - 'AWL'    # Always Warm
  - 'ACL'    # Always Cold
  - 'SVMR'   # SVM RBF
  - 'ONL'    # Online
  - 'LR'     # Logistic Regression
  - 'SVML'   # SVM Linear
  - 'RF'     # Random Forest
  - 'DCT'    # Decision Tree
  - 'KNN'    # K-Nearest Neighbors
  - 'SV-Grid' # SVM with Grid Search
  - 'HV'     # Histogram Voting
  - 'SV'     # Soft Voting

# Cost weight configuration (single value or list)
cost_weight: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
```

## Usage

### Basic Execution

Run the framework with default configuration:

```bash
python main.py
```

### Custom Configuration

Specify a custom configuration file:

```bash
python main.py --config config/custom_config.yaml
```

### Cost Weight Analysis

The framework supports cost-weight analysis to balance cost and latency:

- **Low cost_weight (0.1-0.3)**: Favors latency, more aggressive in classifying as HOT
- **Medium cost_weight (0.4-0.6)**: Balanced between cost and latency
- **High cost_weight (0.7-1.0)**: Favors cost, more conservative in classifying as HOT

## Models

### Machine Learning Models

- **SVMR**: Support Vector Machine with RBF kernel
- **SVML**: Support Vector Machine with Linear kernel
- **RF**: Random Forest
- **KNN**: K-Nearest Neighbors
- **DCT**: Decision Tree
- **LR**: Logistic Regression
- **SV-Grid**: SVM with Grid Search optimization
- **HV**: Histogram Voting (ensemble)
- **SV**: Soft Voting (ensemble)

### Baseline Strategies

- **AHL**: Always Hot - Classifies all objects as HOT
- **AWL**: Always Warm - Classifies all objects as WARM
- **ACL**: Always Cold - Classifies all objects as COLD
- **ONL**: Online - Makes decisions based on current access patterns

## Output

The framework generates comprehensive results in the `results/` directory:

### CSV Results
- `resultados_finais.csv`: Detailed results for all models and configurations

### Analysis Plots
- Accuracy, precision, recall, and F1-score trends
- Cost and latency analysis
- Pareto frontier plots
- Sensitivity analysis heatmaps

### Metrics Included
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Cost Metrics**: Total cost, oracle cost comparison
- **Latency Metrics**: Total latency, oracle latency comparison
- **Confusion Matrix**: Detailed classification breakdown

## Analysis Tools

### Sensitivity Analysis
```bash
python analysis/sensitivity_analysis.py
```

### Trend Visualization
```bash
python analysis/plot_trends.py
```

### Pareto Frontier Analysis
```bash
python analysis/pareto_plot.py
```

## Cost Model

The framework implements a comprehensive cost model considering:

### Storage Costs
- **HOT**: $0.0400 per GB
- **WARM**: $0.0150 per GB
- **COLD**: $0.0040 per GB

### Operation Costs
- **HOT**: $0.0004 per 1K operations
- **WARM**: $0.0010 per 1K operations
- **COLD**: $0.0020 per 1K operations

### Retrieval Costs
- **HOT**: $0.0060 per GB
- **WARM**: $0.0110 per GB
- **COLD**: $0.0200 per GB

### Latency Model
- **HOT**: 400ms
- **WARM**: 800ms
- **COLD**: 1200ms

## Data Format

### Access Data
CSV format with columns representing weeks and rows representing objects:
```
NameSpace,Week1,Week2,Week3,...
Object1,5,3,0,...
Object2,0,1,2,...
```

### Volume Data
CSV format with object volumes:
```
NameSpace,Volume_GB
Object1,1.5
Object2,0.8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

[Add contact information here]

## Acknowledgments

[Add acknowledgments here]
