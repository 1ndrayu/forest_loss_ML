#!/usr/bin/env python3
"""
Forest Loss Prediction - Inference Script
==========================================

A simple script for making predictions on new data using a trained model.

Usage:
    python inference.py --input new_data.csv --output predictions.csv
    python inference.py --input new_data.csv --output predictions.csv --model custom_model.pkl
"""

import argparse
import pandas as pd
import sys
import os
from forest_loss_prediction import ForestLossPredictor

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="Make forest loss predictions on new data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py --input new_data.csv --output predictions.csv
    python inference.py --input new_data.csv --output predictions.csv --model custom_model.pkl
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file with features (same schema as training data)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file for predictions'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='forest_loss_model.pkl',
        help='Path to trained model file (default: forest_loss_model.pkl)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train a model first or specify the correct model path.")
        sys.exit(1)
    
    try:
        if args.verbose:
            print("=== Forest Loss Prediction Inference ===")
            print(f"Input file: {args.input}")
            print(f"Output file: {args.output}")
            print(f"Model file: {args.model}")
            print()
        
        # Initialize predictor
        if args.verbose:
            print("Loading model...")
        
        predictor = ForestLossPredictor()
        predictor.load_model(args.model)
        
        if args.verbose:
            print("Model loaded successfully!")
            print("Making predictions...")
        
        # Make predictions
        results = predictor.predict_new_data(args.input)
        
        if results is not None:
            # Save results
            results.to_csv(args.output, index=False)
            
            if args.verbose:
                print(f"Predictions completed for {len(results)} samples")
                print(f"Average predicted probability: {results['predicted_probability'].mean():.4f}")
                print(f"Predicted forest loss cases: {results['predicted_forest_loss'].sum()}")
                print(f"Results saved to: {args.output}")
            else:
                print(f"Predictions completed: {len(results)} samples, {results['predicted_forest_loss'].sum()} predicted losses")
                print(f"Results saved to: {args.output}")
        else:
            print("Error: Failed to generate predictions!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
