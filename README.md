# An-Exploration-of-Sampling-Stratigies-for-Diffusion-Large-Language-Models

## Fall 2025 Data Science Capstone Project

This is my data science capstone project for my undergraduate studies at NYU Shanghai

### Abstract
Diffusion-based Large Language models have emerged as promising alternative to current autoregressive large language models. They have shown promising potentials in achieving global coherence in text generation, and more related to this work, lowering latency with faster inference due to its parallel generation nature. Such potential of faster inference is far from being fully realized and is still in need of further exploration as diffusion large language model is still a relatively new and developing field. Sampling strategies are an important approach to inference optimization. This project tries to realize such potential of faster inference through an exploration of various selected sampling strate- gies for diffusion large language models. Variants of several typical sampling strategies suited for diffusion large language models have been selected and investigated. This project also applies a variant of speculative decoding, self-speculative decoding, which at the time of this project, has been under explored and in-sufficiently implemented. A comparison and analysis of the sampling strategies is also given. This project demonstrates that sampling strategies are an effective approach to optimizing diffusion language model inference, and provides valuable insights to future studies on using sampling strategies to optimize diffusion language model inference.

## See my Report here: 
- 📄 **Project Report**: [Fall-2025 Data Science Project Report: An Exploration of Sampling Strategies for Diffusion Large Language Models](Project-Report/Capstone_Project_Report.pdf)

## Sampling Strategies Explored:  In this project, we explore the following sampling strategies (and optimization) for dLLMs

### [Adative Parallel Decoding](https://github.com/danielmisrael/apd/tree/main)

### [Slow-Fast Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling/tree/main)

### [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)

### [Self-Speculative Decoding for dLLM](https://arxiv.org/abs/2510.04147)

### [LLaDA Baseline](https://github.com/ML-GSAI/LLaDA)

## Access
The code for the different sampling strategies is included in [Sampling_strategies_for_dLLMs](Sampling_strategies_for_dLLMs/), access and run according to each strategy
