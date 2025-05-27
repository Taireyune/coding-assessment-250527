"""
Allow model to be executed as script in a terminal.
Take config file and data file as arguments.
"""
import argparse
import model
import sys

def main():
    parser = argparse.ArgumentParser(description="Run the Experiment Classifier model.")
    parser.add_argument(
        "--config-dir", required=True, type=str, help="Path to the model configuration file."
    )
    parser.add_argument(
        "--data-dir", required=True, type=str, help="Path to the data file containing experiment names and time series."
    )
    parser.add_argument(
        "--destination", 
        default="results.txt",
        type=str, help="Path to save the result."
    )
    args = parser.parse_args()

    configs = model.load_model_config(args.config_dir)
    try:
        names, data = model.format_input(open(args.data_dir, "r").read())
    except ValueError as e:
        print(f"Error in input data format: {e}")
        sys.exit(1)

    classifier = model.ExperimentClassifier(configs)
    results = []
    for i, ts in enumerate(data):
        try:
            results.append(classifier(ts))
        except Exception as e:
            print(f"Error processing time series {i}, experiment {names[i]}: {e}")
            sys.exit(1)

    with open(args.destination, "w") as f:
        f.write(model.format_output(names, results))


if __name__ == "__main__":
    main()