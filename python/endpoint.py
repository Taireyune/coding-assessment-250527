"""
FastAPI endpoint for the Experiment Classifier model.
"""

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import argparse

import model


def setup():
    parser = argparse.ArgumentParser(description="Run the Experiment Classifier model.")
    parser.add_argument(
        "--config-dir", required=True, 
        type=str, help="Path to the model configuration file."
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        type=str,
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", default=True, action="store_true")
    args = parser.parse_args()

    configs = model.load_model_config(args.config_dir)
    classifier = model.ExperimentClassifier(configs)
    return classifier, args

classifier, args = setup()


class InputData(BaseModel):
    data: str


class ResponseData(BaseModel):
    results: str


app = FastAPI()


@app.post("/", response_model=ResponseData)
async def classify_experiment(data: InputData):
    """
    Endpoint to classify experiments based on time series data.
    
    Args:
    - data (ExperimentData): Input data containing experiment names and time series.
    
    Returns:
    - dict: Classification results for each experiment.
    """
    try:
        # Assuming model is already imported and configured
        names, time_series = model.format_input(data.data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    results = []
    for i, ts in enumerate(time_series):
        try:
            results.append(classifier(ts))
        except Exception as e:
            raise HTTPException(
                status_code=422, 
                detail=f"Error processing time series {i}, experiment {names[i]}: {str(e)}"
            )
        
    return ResponseData(results=model.format_output(names, results))


if __name__ == "__main__":
    uvicorn.run("endpoint:app", host=args.host, port=args.port, reload=args.reload)

