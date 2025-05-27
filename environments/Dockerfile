FROM python:3.12-bullseye

WORKDIR /app
COPY environments/requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY python/endpoint.py ./endpoint.py
COPY python/model.py ./model.py
COPY configs/model-configs-1-0-0.json ./configs.json

CMD ["python", "endpoint.py", "--config", "configs.json", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000