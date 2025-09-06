# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

## 📋 Basic Information
- **Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Year**: 2023
- **Venue**: ICLR 2023
- **Link**: [arXiv](https://arxiv.org/abs/2210.17323)



# 1. 📖 Paper Understanding

- first method to show accurate 4-bit PTQ on LLMs up to 175B parameters.
- It proved that LLMs can be quantized without retraining and still retain performance close to FP16.
- Enabled wide deployment of large models on consumer GPUs
- Became the foundation for many open-source inference stacks

## The Problem

### What problem does this paper solve?

LLM suffer from high computation and storage cost. Specifically:

- Even inference for large GPT models requires multiple performant GPUs due to massive size
- This severely limits the usability and accessibility of such models
- Existing model compression techniques have limited applicability and performance at the scale and complexity of GPT models
- Need for efficient quantization methods that preserve accuracy while enabling deployment on more accessible hardware

### Prior art and why they didn't work well:

Existing post-training quantization methods had significant limitations:
- Previous one-shot quantization methods achieved limited compression gains
- Accuracy degradation was substantial when applied to very large models
- Quantization methods were not optimized for transformer architectures
- Computational requirements for quantization itself were prohibitive for billion-parameter models


## The Key Idea

A new one-shot weight quantization method based on approximate second-order information that is both highly accurate and highly efficient.

Key innovations:
- **One-shot quantization**: No need for retraining or fine-tuning after quantization
- **Second-order information**: Uses approximate Hessian information for more accurate quantization decisions
- **Layer-wise quantization**: Processes one transformer layer at a time to manage memory requirements

Intuition Behind Why It Works:
- second-order statistics (Hessian) propagate and correct quantization error as it goes → minimizes output distortion globally
- Works especially well for transformers where linear projections dominate computation

Advantages:
- **Speed**: Can quantize 175B parameter models in approximately 4 GPU hours
- **Accuracy**: Negligible accuracy degradation compared to uncompressed baseline
- **Efficiency**: More than doubles compression gains relative to previous one-shot methods
- **Deployment**: Enables 175B parameter model execution on single GPU for the first time

## The Challenge
### What are the main challenges in solving this problem?

- **Scale complexity**: Quantizing models with 175+ billion parameters requires managing enormous computational and memory requirements
- **Accuracy preservation**: Maintaining model performance when reducing precision from 16-bit to 3-4 bits per weight
- **Computational efficiency**: Quantization process itself must be fast enough to be practical
- **Second-order optimization**: Computing and utilizing second-order information (Hessian) for billions of parameters is computationally prohibitive
- **Hardware deployment**: Ensuring quantized models can actually run efficiently on target hardware

## The Method
### Brief overview (detailed analysis in Section 2):

Apply OBQ with additional system optimizations.

- **Layer-by-layer processing**: Quantizes one transformer layer at a time to manage memory
- **Approximate second-order information**: Uses diagonal approximation and other techniques to make Hessian computation tractable
- **Optimal Brain Quantization (OBQ) framework**: Extends OBQ method with computational optimizations for large scale
- **Greedy quantization**: Quantizes weights one at a time in order of importance
- **Error compensation**: Adjusts remaining weights to compensate for quantization errors
- **Calibration dataset**: Uses small calibration set (no training data required) to compute statistics

## Strengths:
- **Exceptional speed**: 4 GPU hours for 175B parameters vs. days/weeks for alternative methods
- **High compression**: Achieves 3-4x compression (4-bit) with minimal accuracy loss
- **Single GPU deployment**: Enables large model inference on single GPU for first time
- **No retraining**: Post-training method requires no additional training data or compute
- **Hardware speedups**: Delivers 3.25x speedup on A100, 4.5x on A6000
- **Extreme quantization**: Can work reasonably even at 2-bit or ternary levels
- **Scalable**: Demonstrated to work on models up to 175B parameters

## Impact & Contributions
### Key contributions to the field:

- **Democratization of large models**: Enabled deployment of 175B parameter models on single GPUs
- **Quantization methodology**: Established new standard for post-training quantization of large language models
- **Second-order optimization**: Showed how to make second-order methods practical for billion-parameter models
- **Performance benchmarking**: Demonstrated significant inference speedups on practical hardware
- **Open source implementation**: Made tools available to broader research community

### How did this paper change the field after its release?

- **Industry adoption**: GPTQ became widely adopted in industry for deploying large language models
- **Research acceleration**: Enabled researchers with limited hardware to experiment with large models
- **Tool ecosystem**: Spawned ecosystem of tools and libraries for quantized inference
- **Hardware co-design**: Influenced development of hardware optimized for low-precision inference
- **Follow-up research**: Inspired numerous improvements and extensions to quantization methods
- **Accessibility**: Significantly lowered barrier to entry for working with large language models



### Key Insights for Modern AI Systems

- **Scale advantages**: Larger models often quantize better than smaller ones
- **Second-order information**: Curvature-based methods significantly outperform naive approaches  
- **Calibration efficiency**: Small, representative datasets sufficient for high-quality quantization
- **Layer heterogeneity**: Different network layers have very different quantization sensitivities
- **Hardware considerations**: Quantization benefits depend heavily on having optimized inference kernels



# 2. 🔬 Key Technical Details

## Method

### Core Algorithm: Layer-wise Quantization with Second-order Information

**Input**: Pre-trained transformer model, calibration dataset, target bit-width
**Output**: Quantized model with minimal accuracy degradation

**Key components:**
- **Hessian approximation**: Computes approximate second-order information efficiently
- **Greedy quantization**: Quantizes weights one at a time based on importance
- **Error propagation**: Updates remaining weights to compensate for quantization errors
- **Layer-wise processing**: Handles memory constraints by processing layers sequentially


### Arbitrary Order Insight
In llm, the order in which to pick the greedy quantization doesn't matter because:
1. each layer is heavily over-parameterized anyways
2. the compensation mechanism spreads error

So using the same fixed column order for all rows works almost as well.
- That means you only need to do the Hessian inverse update once per column, not separately for every row.

complexity goes from:
OBQ: $$O(d_{row} \cdot d_{col}^3)$$
GPTQ: $$O(max(d_{row} \cdot d_{col}^2, d_{col}^3))$$

### Block-wise computation 
Work on smaller block to fit into the cache line instead of working on the whole large matrix at once.

Update the weight updates and hassian inverse matrix in block fashion as well.



### Cholesky Reformulation

First two steps creates numeric instability on very large model, because repeated block updates cause numerical errors to accumlates. Roundoff errors loss the positive-definiteness of the H matrix property. This can call weight update to go to the wrong direction due to overflow. Thus quantization collapse. And this always happen for LLM > few billion parameters.

Idea: for each block of weight, it only need the row corresponding to the weight being quantized (starting at the diagnonal element of H). So rather than maintaining the entire H^-1 matrix, we can compute the necessary row in a numerically stable way on the fly for that row we are working on.
$$
H = 2 X X^T + \lambda I
$$

lambda is a small dampening factor. A small positive constant that is ~1% of the average diagonal of H
- add numerical stability
- a small bias in case H is nearly singular (small eigenvalues)
- It’s a deterministic shrinkage of the Hessian, applied uniformly to all diagonal entries.

Instead of repeatedly mutating H^{-1}, GPTQ computes the Cholesky once and uses it to fetch the needed inverse information as quantization proceeds.

With Cholesky, you don’t keep the whole H^-1; you store a compact factorized copy that never drifts, and when you need a number you compute it stably from that.


### Mathematical Formulation

**Optimal Brain Quantization (OBQ) Extension**:
- Objective: minimize ||WX - W_quantized X||²_F over calibration data X
- Use second-order Taylor expansion: ΔL ≈ g^T Δw + ½ Δw^T H Δw  
- Greedy solution: quantize weight with smallest increase in loss function
- Error compensation: update remaining weights using H^(-1) to minimize residual error

**Computational Optimizations**:
- Diagonal Hessian approximation: H ≈ diag(h₁, h₂, ..., hₙ)
- Lazy weight updates: batch updates to reduce computational overhead
- Memory management: process layers sequentially to fit in GPU memory


### Key Algorithms and Techniques

1. **Adaptive Quantization**:
   - Different layers may use different bit-widths based on sensitivity analysis
   - Attention layers often more sensitive than feed-forward layers
   - Can mix 4-bit, 3-bit, and even 2-bit quantization within single model

2. **Calibration Data Selection**:
   - Use small subset (128-1024 samples) of training or validation data
   - Representative sampling important for capturing model behavior
   - No gradient computation required - only forward passes

3. **Inference Optimization**:
   - Custom CUDA kernels for efficient quantized matrix multiplication
   - Memory layout optimization for quantized weights
   - Mixed-precision computation maintaining accuracy in accumulation

## Interesting Findings and Insights

- **Scaling behavior**: Larger models are actually easier to quantize with minimal accuracy loss
- **Layer sensitivity**: Different transformer layers have very different sensitivity to quantization
- **Calibration efficiency**: Very small calibration sets (128 samples) sufficient for good results
- **Hardware impact**: Memory bandwidth becomes primary bottleneck rather than compute
- **Extreme quantization**: Even 2-bit quantization can work reasonably for some applications

## Experimental Results

### Performance Metrics:
- **Compression**: 4x reduction in model size (16-bit to 4-bit)
- **Speed**: 3.25x inference speedup on NVIDIA A100
- **Speed**: 4.5x inference speedup on NVIDIA A6000  
- **Accuracy**: <1% degradation on most language modeling benchmarks
- **Quantization time**: 4 GPU hours for 175B parameter OPT model

### Model Coverage:
- Successfully applied to GPT models up to 175B parameters
- Works across different model families (GPT, OPT, BLOOM)
- Maintains performance across diverse downstream tasks
- Enables single-GPU inference for models previously requiring multiple GPUs


# 📚 References

- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. *ICLR 2023*.
- Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., Van Baalen, M., & Blankevoort, T. (2021). A white paper on neural network quantization. *arXiv preprint*.
- Hassibi, B., & Stork, D. G. (1993). Second order derivatives for network pruning: Optimal brain surgeon. *Advances in Neural Information Processing Systems*.
- LeCun, Y., Denker, J. S., & Solla, S. A. (1990). Optimal brain damage. *Advances in Neural Information Processing Systems*.

## Background Concepts: Neural Network Quantization

Quantization in deep learning:
- reduce the precision of weights or sometime activations
- 16 or 32 bit floats -> 8 or 4 bits
- This saves **memory** and improves **inference speed** less byte to move through memory -> reduce memory bandwidth bottleneck



### Historical Context

1. **Early Quantization Work (1990s-2000s)**:
   - Initial focus on reducing memory and computation for deployment
   - Binary neural networks and extreme quantization experiments  
   - Limited to smaller networks due to accuracy degradation

2. **Deep Learning Era (2010s)**:
   - Quantization became critical for mobile and edge deployment
   - Training-time quantization methods (QAT) became popular
        - retrain model with quantization simulated. Works well but too expensive for multi-billion-parameter models.
   - Post-training quantization emerged for pre-trained models
        - quantize without retraining, faster but less accurate. Many PTQ methods fail for LLMs, especially at 4-bit.

3. **Large Language Model Era (2020s)**:
   - Model sizes exploded beyond what could fit on single GPUs
   - Post-training methods became essential due to retraining costs
   - Need for high-quality quantization without access to training data

### Quantization Fundamentals

1. **Uniform Quantization**:
   - Maps continuous weight values to discrete levels
   - Q(w) = round((w - z) / s) where s is scale, z is zero-point
   - Linear mapping between quantized and full-precision values

2. **Post-Training Quantization (PTQ)**:
   - Quantizes pre-trained model without additional training
   - Uses calibration data to determine optimal quantization parameters
   - Much faster than quantization-aware training (QAT)

3. **Second-Order Methods**:
   - Use curvature information (Hessian) to make better quantization decisions
   - Can identify which weights are most/least important for model accuracy
   - More computationally expensive but often yields better results


## Useful Resources
- Explaination: https://www.youtube.com/watch?v=6J_0BDqMFi0 