#!/usr/bin/env python3
"""
Signal Evaluation and Labeling

This script evaluates pre-generated trading signals from gold_train.csv and labels them as
'Successful', 'Unsuccessful', or 'Neutral' based on subsequent price movements.
"""

import pandas as pd
import numpy as np

def main():
    # Define file paths
    input_csv = 'data/gold_train.csv'
    output_csv = 'gold_train_labeled.csv'

    # Load the dataset
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} rows from {input_csv}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        return
    except Exception as e:
        print(f"Error loading CSV {input_csv}: {e}")
        return

    # Parse datetime and sort
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Verify required columns
    required_cols = ['close', 'Signal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return

    # Calculate price changes
    df['next_close'] = df['close'].shift(-1)
    df['price_diff'] = df['next_close'] - df['close']

    # Initialize strategy_success column
    df['strategy_success'] = np.nan

    # Evaluate BUY signals (Signal == 1)
    buy_mask = df['Signal'] == 1
    df.loc[buy_mask, 'strategy_success'] = df.loc[buy_mask, 'price_diff'] > 0

    # Evaluate SELL signals (Signal == -1)
    sell_mask = df['Signal'] == -1
    df.loc[sell_mask, 'strategy_success'] = df.loc[sell_mask, 'price_diff'] < 0

    # Mark HOLD signals (Signal == 0) as Neutral
    hold_mask = df['Signal'] == 0
    df.loc[hold_mask, 'strategy_success'] = pd.NA

    # Handle the last row (no next price)
    last_row_mask = df['next_close'].isna()
    df.loc[last_row_mask, 'strategy_success'] = pd.NA

    # Convert to readable labels
    success_map = {True: 'Successful', False: 'Unsuccessful', pd.NA: 'Neutral'}
    df['strategy_success'] = df['strategy_success'].map(success_map).fillna('Neutral')

    # Display results
    print("\nSignal Success Distribution:")
    print(df['strategy_success'].value_counts())

    print("\nExample rows with strategy evaluation:")
    print(df[['datetime', 'close', 'Signal', 'next_close', 'price_diff', 'strategy_success']].head(10))

    # Calculate success rates
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    hold_signals = df[df['Signal'] == 0]

    print("\nSignal Distribution:")
    print(f"Total signals: {len(df)}")
    print(f"BUY signals: {len(buy_signals)}")
    print(f"SELL signals: {len(sell_signals)}")
    print(f"HOLD signals: {len(hold_signals)}")

    print("\nSuccess Rates:")
    print(f"BUY success rate: {(buy_signals['strategy_success'] == 'Successful').mean():.2%}")
    print(f"SELL success rate: {(sell_signals['strategy_success'] == 'Successful').mean():.2%}")

    # Remove temporary columns before saving
    df_to_save = df.drop(columns=['next_close', 'price_diff'])

    # Save to CSV
    try:
        df_to_save.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved labeled data to '{output_csv}'")
    except Exception as e:
        print(f"\nError saving file '{output_csv}': {e}")

if __name__ == "__main__":
    main() 