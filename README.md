# Study for for Berkeley AISYS PhD Prelim

This repository contains my notes and insights from studying for the AISYS prelim at berkeley.

## 📚 Reading List

The notes are organized according to the [official AISYS prelim reading list](https://learning-systems.notion.site/0bd2bf6cf59e4485b65d2bef84352f26?v=48c7b12c9e9d45ecb05c70b6504dc999).

## 📖 Paper Notes

| # | Paper Title | Category | Venue | Year | PDF Link | Notes |
|---|-------------|----------|-------|------|----------|-------|
| 1 | ImageNet Classification with Deep Convolutional Neural Networks | Training Systems | NeurIPS | 2012 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | [📝](alexnet.md) |
| 2 | XGBoost: A Scalable Tree Boosting System | ClassicML | KDD | 2016 | [PDF](https://arxiv.org/pdf/1603.02754) | [📝](paper_02.md) |
| 3 | Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization | TensorOptimization | - | - | [PDF](https://arxiv.org/abs/1910.02653) | [📝](paper_03.md) |
| 4 | Anatomy of High-Performance Matrix Multiplication | Architecture | - | - | [PDF](https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf) | [📝](paper_04.md) |
| 5 | Why Systolic Architectures | Architecture | - | - | [PDF](http://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf) | [📝](paper_05.md) |
| 6 | PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs | Graph Systems | OSDI | 2012 | [PDF](https://www.usenix.org/system/files/conference/osdi12/osdi12-final-167.pdf) | [📝](paper_06.md) |
| 7 | The Case for Learned Index Structures | AI for Systems | SIGMOD | 2018 | [PDF](https://arxiv.org/abs/1712.01208) | [📝](paper_07.md) |
| 8 | Using the BSP cost model to optimise parallel neural network training | Pre-rec, Scaling Training | Future Generation Computer Systems | 1998 | [PDF](https://www.sciencedirect.com/science/article/abs/pii/S0167739X98000430) | [📝](paper_08.md) |
| 9 | Machine Learning: The High Interest Credit Card of Technical Debt | Industry Perspective | NeurIPS Workshop | 2014 | [PDF](https://research.google/pubs/machine-learning-the-high-interest-credit-card-of-technical-debt/) | [📝](paper_09.md) |
| 10 | Roofline: An Insightful Visual Performance Model for Multicore Architectures | Architecture | CACM | 2009 | [PDF](https://dl.acm.org/doi/10.1145/1498765.1498785) | [📝](paper_10.md) |
| 11 | AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving | Inference Systems | OSDI | 2023 | [PDF](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan) | [📝](paper_11.md) |
| 12 | TensorFlow: A System for Large-Scale Machine Learning | Training Systems | OSDI | 2016 | [PDF](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf) | [📝](paper_12.md) |
| 13 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Quantization | - | 2022 | [PDF](https://arxiv.org/abs/2210.17323) | [📝](paper_13.md) |
| 14 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | Scaling Training, Training Systems | - | 2018 | [PDF](https://arxiv.org/abs/1909.08053) | [📝](paper_14.md) |
| 15 | Efficient Memory Management for Large Language Model Serving with PagedAttention | Inference Systems | SOSP | 2023 | [PDF](https://arxiv.org/abs/2309.06180) | [📝](paper_15.md) |


## 📝 Note Structure

Each paper note follows a consistent format with two main sections:

```markdown
# Paper Title


## 1. 📖 Paper Understanding

### The Problem
- **What problem does this paper solve?**
- **Prior art and why they didn't work well:**
- **Related work:**

### The Challenge
- **What are the main challenges in solving this problem?**

### The Key Idea
- **High-level approach to solving the problem:**

### The Method
- **Brief overview (detailed analysis in Section 2):**

### Pros & Cons
- **Strengths:**
- **Weaknesses/Limitations:**

### Impact & Contributions
- **Key contributions to the field:**
- **How did this paper change the field after its release?**

### [Optional] Background & History
- **Useful background knowledge:**
- **Pre-history and context:**

## 2. 🔬 Key Technical Details

### Deep Method Understanding
- **How does the method work in detail?**
- **Key algorithms and techniques:**

### Case Studies & Examples
- **Specific examples and implementations:**

### Technical Insights
- **Deep technical understanding and nuances:**

```

## 🤝 Contributing

Feel free to:
- Open issues for corrections or suggestions
- Open issues for open discussion on the paper
- Submit pull requests for improvements
- Use these notes for your own studies
- Share with fellow students


**Note**: This repository is a work in progress. Papers will be added and notes will be updated as I progress through the ready list.

**Acknowledgement**: Creating this repo is inspired by Shu's System study note: [Berkeley OS Prelim Notes](https://github.com/lynnliu030/berkeley-os-prelim)
