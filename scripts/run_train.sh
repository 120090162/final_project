#!/bin/bash

# Export the required environment variable
export JAX_DEFAULT_MATMUL_PRECISION=highest

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the training script using the absolute path, passing any arguments
python "$DIR/train_jax_ppo.py" "$@"
