#!/bin/bash
# Run this after activating conda environment
dvc init
dvc remote add -d local /tmp/dvc-storage
echo "✅ DVC initialized"
