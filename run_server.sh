#!/bin/bash
# This is a script to run the Flask server with the deployed model
cd flask_apps
export FLASK_APP=predict_app.py
python -m flask run --host=0.0.0.0
cd ..
