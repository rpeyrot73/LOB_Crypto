import argparse
from data_pipeline import run_data_pipeline
from training_pipeline import run_training_pipeline
from backtest import run_pipeline_backtest

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run different pipelines for the project.")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["data", "training", "backtest"],
        required=True,
        help="Specify which pipeline to run: 'data', 'training', or 'backtest'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file. Default is 'config2.json'."
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Specify this flag to enable training in the training pipeline."
    )
    parser.add_argument(
        "--run_backtest",
        action="store_true",
        help="Specify this flag to enable backtesting in the backtest pipeline."
    )
    args = parser.parse_args()

    # Run the specified pipeline
    if args.pipeline == "data":
        print(f"Running data pipeline with config: {args.config}")
        run_data_pipeline(args.config)
    elif args.pipeline == "training":
        print(f"Running training pipeline with config: {args.config}")
        run_training_pipeline(args.config, training=args.training)
    elif args.pipeline == "backtest":
        print(f"Running backtest pipeline with config: {args.config}")
        run_pipeline_backtest(args.config, run_backtest=args.run_backtest)

if __name__ == "__main__":
    main()
