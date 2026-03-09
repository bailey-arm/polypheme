#!/bin/bash
set -e

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name polypheme --display-name "polypheme"

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo "Launch notebooks with: jupyter lab"
