#!/bin/bash
# Clear Python cache files to force reimport of updated modules

echo "Clearing Python cache files..."

# Remove __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Remove .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null

# Remove .pyo files
find . -type f -name "*.pyo" -delete 2>/dev/null

echo "✓ Python cache cleared successfully"
echo ""
echo "Please run your command again:"
echo "  python scripts/train_pipeline.py --config config.yaml"
