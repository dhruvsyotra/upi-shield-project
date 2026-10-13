#!/usr/bin/env bash
set -e
python scripts/data_generator.py
python scripts/train_model.py
uvicorn main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID' EXIT
sleep 3
streamlit run app.py
