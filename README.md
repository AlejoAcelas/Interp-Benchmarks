# Interpretability Benchmarks  

This repository provides detailed explainability analyses for Transformer models trained on algorithmic tasks. The explanations map model activations to simplified causal graphs that recover over 90% of the original model loss. This makes them valuable benchmarks for evaluating interpretability techniques.  

## Contents  

The repository assumes basic knowledge of interpretability methods like [causal scrubbing](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) and [circuit-style analysis](https://arxiv.org/abs/2211.00593) of Transformer models. Key items:

* Explanations formatted as simplified causal models
* Resampling tests quantify explanation accuracy  
* Matches or exceeds the accuracy of [previous attemps](https://www.alignmentforum.org/posts/kjudfaQazMmC74SbF/causal-scrubbing-results-on-a-paren-balance-checker) at causal scrubbing analysis on algorithmic tasks
* Ideal for benchmarking new interpretability techniques
* Notebooks and scripts for training, evaluation, and analysis
* Modular codebase for extending to new models and tasks

By open sourcing detailed analyses tied to model performance, this repository aims to advance interpretability research. Contributions welcome!
