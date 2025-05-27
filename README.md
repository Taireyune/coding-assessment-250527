# LGC Data Scientist Hiring – Code Challenge

## Notebook Setup

Setup Jupyter Notebook locally using these instructions. Start from the project root folder:

1. Install python3.12. Check python installation using `python3 --version`

2. Setup venv: run  `python3 -m venv .venv-notebook`

3. Activate environment:
       - for mac.  `source .venv-notebook/bin/activate` then check using `which python3`
       - for windows: `.venv-notebook\Scripts\activate.bat` then check using `where python3`

4. Install dependencies using `pip3 install environments/requirements-notebook.txt`

5. Run spinup jupyter notebook using `jupyter notebook`. Access the notebook at http://localhost:8888

## Inference on local file
 
1. Install python3.12. Check python installation using `python3 --version`

2. Setup venv: run  `python3 -m venv .venv-deploy`

3. Activate environment:
       - for mac.  `source .venv-deploy/bin/activate` then check using `which python3`
       - for windows: `.venv-deploy\Scripts\activate.bat` then check using `where python3`

4. Install dependencies using `pip3 install environments/requirements-deploy.txt`

5. Go to the python folder. `cd python`

5. Run this in the terminal:
```
python main.py --config-dir "../configs/model-configs-1-0-0.json" --data-dir "../datasets/evaluation_data_external.csv" --destination "../main-results.csv"
```
    Adjust the path parameters as needed.

## Deploy for scalability

The following is instructions on spinning the API endpoint as a docker container.

1. Install Docker. Check installation using `docker --version`

2. Navigate to project root folder. Build container with the following:
```
docker build -t coding-challenge:1.0.0 -f environments/Dockerfile .
```

3. Run the container on port 8000 using
```
docker run -it --rm -p 8000:8000 coding-challenge:1.0.0 
```

4. Go to http://localhost:8000/docs to interact with the endpoint

5. Navigate to POST request dropdown, use the "Try it out" button

6. Paste the content of evaluation data CSV into "string" with newline characters "\n" included

7. Click on "Execute" to inspect inference response.
