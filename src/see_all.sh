#!/bin/bash

# This script is intended to be used for displaying all test JSON files
# Directory to search for JSON files
SEARCH_DIR="./src/test_data"

# Check if an argument is given
if [ "$#" -eq 1 ]; then
  SEARCH_DIR=$1
fi

# Find all JSON files in the directory and its subdirectories
# and open them with labelme_draw_json
find "$SEARCH_DIR" -maxdepth 1 -type f -name '*dumped.json' | while read -r file; do
  echo "Opening '$file'..."
  labelme_draw_json "$file"
done
