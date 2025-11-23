#!/bin/bash
echo "Starting MLflow UI on port 5001..."
PYTHONWARNINGS="ignore" mlflow ui --host 0.0.0.0 --port 5001 > /dev/null 2>&1 &
echo "MLflow started! Open http://localhost:5001"
echo "To stop: pkill -f 'mlflow ui'"
