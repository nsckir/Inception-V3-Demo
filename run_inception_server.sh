#!/bin/bash

# Load, freeze, optimize, and export ResNet model
if [ ! -f serving/static/inception_v3_frozen.pb ]; then
    echo "Creating frozen ResNet model file.."
    python3 inception_v3_export.py
    echo "Done"
fi

# Create temp directory if necessary
mkdir -p serving/temp

# Serve Flask application
echo "Launching server."
cd serving
export FLASK_APP=serving_inception.py

