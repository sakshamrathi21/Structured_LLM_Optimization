#!/bin/bash
set -e

echo "Building and installing chunked_hash_tree C++ extension..."

# Navigate to the extension directory
cd /home/preeti/sglang_dir/sglang/python/sglang/srt/managers/chunked_hash_tree_ext

# Install required system packages if not present
# This assumes Ubuntu/Debian based system
sudo apt-get update
sudo apt-get install -y build-essential libxxhash-dev

# Ensure pybind11 is installed in the python environment
pip install pybind11

# Build and install the extension
pip install --no-build-isolation -e .

echo "Installation complete! You can now test importing the module:"
python -c "from chunked_hash_tree import ChunkedHashTree_RL; print('Successfully imported ChunkedHashTree_RL')"
