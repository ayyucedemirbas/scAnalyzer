#!/bin/bash
echo "Sorting imports..."
isort .
echo "Formatting code..."
black .
echo "Done!"