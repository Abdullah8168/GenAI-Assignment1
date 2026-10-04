# MLflow UI for the experiment-tracking records produced during training (Kaggle).
# The runs store absolute artifact paths from the training machine; fix_mlruns_paths.py
# rewrites them to /mlruns at start-up so logged figures/checkpoints open in the UI.
FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=2.17,<3" pyyaml
COPY tools/fix_mlruns_paths.py /fix_mlruns_paths.py
EXPOSE 5000
CMD python /fix_mlruns_paths.py /mlruns && \
    mlflow ui --backend-store-uri file:///mlruns --host 0.0.0.0 --port 5000
