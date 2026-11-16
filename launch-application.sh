#!/bin/bash


gnome-terminal -- bash -c "source yikes/bin/activate && cd backend/ && uvicorn backend:app --host 127.0.0.1 --reload --port 8000"

gnome-terminal -- bash -c "source yikes/bin/activate && cd backend/ && python batch_retrain.py --daemon --interval-hours 1"

gnome-terminal -- bash -c "cd frontend && npm run dev"


