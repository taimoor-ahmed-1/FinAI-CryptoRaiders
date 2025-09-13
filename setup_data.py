#!/usr/bin/env python3
"""
Data Setup Script for FinRL Pipeline

This script helps users set up the required data structure for the FinRL pipeline.
It creates the necessary directories and provides guidance on data preparation.

Usage:
    python setup_data.py
"""

import os
import sys

def create_directory_structure():
    """Create the required directory structure for the pipeline"""
    
    directories = [
        'data/1sec',
        'data/1min', 
        'data/5min',
        'data/1sec/alpha101',
        'data/1min/alpha101',
        'data/5min/alpha101',
        'output/4_1/1sec',
        'output/4_1/1min',
        'output/4_1/5min',
        'trained_agents/5_7_3/1sec/PPO',
        'trained_agents/5_7_3/1min/PPO',
        'trained_agents/5_7_3/5min/PPO'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\nDirectory structure created successfully!")

def create_sample_data_info():
    """Create a sample data information file"""
    
    info_content = """# Data Requirements

## Required Input Data

To run the FinRL pipeline, you need the following data file:

### Primary Data File
- **File**: `data/BTC_1sec_with_sentiment_risk_train.csv`
- **Description**: 1-second Bitcoin trading data with sentiment and risk scores
- **Size**: ~500MB (varies by time period)
- **Columns**: 
  - `system_time`: Timestamp
  - `midpoint`: Mid-price
  - `spread`: Bid-ask spread
  - `buys`, `sells`: Volume data
  - `bids_distance_*`, `asks_distance_*`: Order book levels (0-14)
  - `bids_notional_*`, `asks_notional_*`: Order book notional values
  - `sentiment_score`: Market sentiment (0-1)
  - `risk_score`: Risk assessment (0-1)

## Data Generation Process

The pipeline will automatically generate the following files:

### Step 1: Data Aggregation
- `data/BTC_1min_with_sentiment_risk_train.csv`
- `data/BTC_5min_with_sentiment_risk_train.csv`

### Step 2: Data Splitting
- `data/1sec/BTC_1sec_with_sentiment_risk_train_1sec_train_70.csv`
- `data/1sec/BTC_1sec_with_sentiment_risk_train_1sec_val_15.csv`
- `data/1sec/BTC_1sec_with_sentiment_risk_train_1sec_test_15.csv`
- (Similar for 1min and 5min timeframes)

### Step 3: Alpha101 Signals
- `data/1sec/alpha101/alpha101_train.npy`
- `data/1sec/alpha101/alpha101_val.npy`
- `data/1sec/alpha101/alpha101_test.npy`
- (Similar for 1min and 5min timeframes)

### Step 4: RNN Predictions
- `output/4_1/1sec/train_predictions.npy`
- `output/4_1/1sec/valid_predictions.npy`
- `output/4_1/1sec/test_predictions.npy`
- (Similar for 1min and 5min timeframes)

### Step 5: Trained Models
- `trained_agents/5_7_3/1sec/PPO/actor.pth`
- `trained_agents/5_7_3/1sec/PPO/critic.pth`
- (Similar for 1min and 5min timeframes)

## Getting Started

1. **Place your data file**: Copy `BTC_1sec_with_sentiment_risk_train.csv` to the `data/` directory
2. **Run the pipeline**: Execute the notebooks in sequence (1 → 2 → 3 → 4 → 5 → 6)
3. **Check results**: Generated files will appear in the respective directories

## Notes

- All generated files are excluded from Git due to size constraints
- The pipeline will create these files automatically when you run the notebooks
- Ensure you have sufficient disk space (~2GB for 1-second data processing)
- GPU is recommended for training steps (4, 5, 6)
"""
    
    with open('data/README.md', 'w') as f:
        f.write(info_content)
    
    print("✓ Created data/README.md with detailed information")

def main():
    """Main setup function"""
    print("🚀 FinRL Pipeline Data Setup")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create data information file
    create_sample_data_info()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Place your BTC_1sec_with_sentiment_risk_train.csv file in the data/ directory")
    print("2. Run the pipeline notebooks in sequence:")
    print("   - 1_data_aggregator.ipynb")
    print("   - 2_data_splitter.ipynb") 
    print("   - 3_alpha_signals_generator.ipynb")
    print("   - 4_rnn_trainer.ipynb")
    print("   - 5_erl_trainer.ipynb")
    print("   - 6_erl_evaluator.ipynb")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()
