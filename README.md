# Interpretability Benchmarks

This repository offers detailed mechanistic explanations for various Transformer models trained on simple algorithmic tasks. Each explanation is formatted as a simplified causal model with nodes mapping to the activations in each trained model. When tested using resampling ablations, these explanations frequently recover over 90% of the original model's loss compared to a random baseline. This makes them highly valuable for benchmarking against other interpretability methods.  

The repository's contents assume basic knowledge of [mechanistic interpretability](https://transformer-circuits.pub/2022/mech-interp-essay/index.html), [causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), and [circuit-style analysis](https://arxiv.org/abs/2211.00593) of Transformer models.

## Repository Structure:

- **docs**:
  - Description of model tasks and identified causal explanations.
  - Description of model training and hyperparameters.
  
- **scripts**: 
  - Exploratory notebooks for each model and task
  - Training scripts

- **src/dataset**: 
  - Data generation and tasks definition
  - Data classification and resampling
  
- **src/experiments**: 
  - Utilities for interpreting models and visualizing activations
  - Causal Scrubbing implementation
  
- **src/train**: 
  - Model definition using `HookedTransformer` from TransformerLens.
  - Training loop  


