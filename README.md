# Interpretability Benchmarks Repository

This repository provides mechanistic interpretations for various Transformer models trained on simple algorithmic tasks. For each task, there are two models: a standard base model and a version with a backdoor implanted during training.

## Repository Structure:

- **docs (in-progress)**:
  - Description of the reversed-engineered algorithm found for each model.
  - Description of model training and hyperparameters.
  
- **src/dataset**: 
  - Classes responsible for generating algorithmic training data for different tasks.
  - Functions used to partition data for causal scrubbing experiments in the `discriminator.py` and `discriminator_utils.py` files.
  
- **src/experiments**: 
  - Helper utilities for model interpretation and activation visualization.
  - Includes the implementation of Causal Scrubbing.
  
- **src/train**: 
  - Model definition using `HookedTransformer` from TransformerLens.
  - Training loop.  

- **scripts**: 
  - Contains scripts used to generate the results and visualizations presented in the docs section.
  - Note: It's still a little messy.

