# Study for for Berkeley AISYS PhD Prelim

This repository contains my notes and insights from studying for the AISYS prelim at berkeley.

## 📚 Reading List

The notes are organized according to the [official AISYS prelim reading list](https://learning-systems.notion.site/0bd2bf6cf59e4485b65d2bef84352f26?v=48c7b12c9e9d45ecb05c70b6504dc999).

## 📖 Paper Notes

| # | Paper Title | PDF Link | Notes | Category | Venue | Year |
|---|-------------|----------|-------|----------|-------|------|
| 1 | ImageNet Classification with Deep Convolutional Neural Networks | [PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | [📝](alexnet.md) | Training Systems | NeurIPS | 2012 |
| 2 | XGBoost: A Scalable Tree Boosting System | [PDF](https://arxiv.org/pdf/1603.02754) | [📝](xgboost.md) | ClassicML | KDD | 2016 |
| 3 | Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization | [PDF](https://arxiv.org/abs/1910.02653) | [📝](checkmate.md) | TensorOptimization | - | - |
| 4 | Anatomy of High-Performance Matrix Multiplication | [PDF](https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf) | [📝](anatomy_mm.md) | Architecture | - | - |
| 5 | Why Systolic Architectures | [PDF](http://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf) | [📝](systolic_architectures.md) | Architecture | - | - |
| 6 | PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs | [PDF](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-167.pdf) | [📝](paper_06.md) | Graph Systems | OSDI | 2012 |
| 7 | The Case for Learned Index Structures | [PDF](https://arxiv.org/abs/1712.01208) | [📝](paper_07.md) | AI for Systems | SIGMOD | 2018 |
| 8 | Using the BSP cost model to optimise parallel neural network training | [PDF](https://www.sciencedirect.com/science/article/abs/pii/S0167739X98000430) | [📝](paper_08.md) | Pre-rec, Scaling Training | Future Generation Computer Systems | 1998 |
| 9 | Machine Learning: The High Interest Credit Card of Technical Debt | [PDF](https://research.google/pubs/machine-learning-the-high-interest-credit-card-of-technical-debt/) | [📝](paper_09.md) | Industry Perspective | NeurIPS Workshop | 2014 |
| 10 | Roofline: An Insightful Visual Performance Model for Multicore Architectures | [PDF](https://dl.acm.org/doi/10.1145/1498765.1498785) | [📝](roofline.md) | Architecture | CACM | 2009 |
| 11 | AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving | [PDF](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan) | [📝](paper_11.md) | Inference Systems | OSDI | 2023 |
| 12 | TensorFlow: A System for Large-Scale Machine Learning | [PDF](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf) | [📝](paper_12.md) | Training Systems | OSDI | 2016 |
| 13 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | [PDF](https://arxiv.org/abs/2210.17323) | [📝](gptq.md) | Quantization | - | 2022 |
| 14 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | [PDF](https://arxiv.org/abs/1909.08053) | [📝](megatron_lm.md) | Scaling Training, Training Systems | - | 2018 |
| 15 | Efficient Memory Management for Large Language Model Serving with PagedAttention | [PDF](https://arxiv.org/abs/2309.06180) | [📝](vllm.md) | Inference Systems | SOSP | 2023 |


## General Knowledge
Hardware architectures and overview: [📝](hardware_architecture.md)

Parallelism:  [📝](parallelism.md)

Transformer:  [📝](transformers.md)


## 🤝 Contributing

Feel free to:
- Open issues for corrections or suggestions
- Open issues for open discussion on the paper
- Submit pull requests for improvements
- Use these notes for your own studies
- Share with fellow students


**Note**: This repository is a work in progress. Papers will be added and notes will be updated as I progress through the ready list.

**Acknowledgement**: Creating this repo is inspired by Shu's System study note: [Berkeley OS Prelim Notes](https://github.com/lynnliu030/berkeley-os-prelim)
