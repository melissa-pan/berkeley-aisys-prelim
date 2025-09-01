# XGBoost: A Scalable Tree Boosting System

## 📋 Basic Information
- **Authors**: Tianqi Chen, Carlos Guestrin
- **Year**: 2016
- **Venue**: KDD
- **Link**: [PDF](https://arxiv.org/pdf/1603.02754)

## 1. 📖 Paper Understanding

### The Problem

> What problem does this paper solve?

> Prior art and why they didn't work well:

> Related work:


### The Key Idea
> High-level approach to solving the problem:



### The Challenge
> What are the main challenges in solving this problem?


### The Method
> Brief overview (detailed analysis in Section 2):


### Pros & Cons
> Strengths:

> Weaknesses/Limitations:

### Impact & Contributions
> Key contributions to the field:



> How did this paper change the field after its release?


### XGBoost Innovations on Gradient Boosting
- **Useful background knowledge:**
  - Gradient boosting and ensemble methods
  - Decision trees and tree learning algorithms
  - Distributed computing and parallel algorithms
  - Cache optimization and memory hierarchy

#### **Pre-history and context:**
- Boosting originated from theoretical work on PAC learning
- AdaBoost (1997) was the first practical boosting algorithm
- Gradient boosting (2001) generalized boosting to arbitrary loss functions
- Random Forest showed power of tree ensembles
- But existing implementations were too slow for big data

#### **Second-order Methods:**
- Traditional gradient boosting uses only first-order gradients
- XGBoost uses second-order (Hessian) information
- Provides better convergence and more accurate approximations

#### **Regularization:**
- Explicit regularization in objective function
- Controls model complexity and prevents overfitting
- L1 and L2 regularization on leaf weights

#### **Systems Optimizations:**
- Focus on making algorithm practically scalable
- Cache-aware algorithms and data structures
- Parallel and distributed computing support

### Impact on Machine Learning

#### **Paradigm Shift:**
- Showed that careful systems engineering could make algorithms practical
- Demonstrated importance of implementation for algorithm adoption
- Influenced design of subsequent ML systems

#### **Competition with Deep Learning:**
- While deep learning dominated unstructured data (images, text)
- XGBoost remained king of structured/tabular data
- Sparked renewed interest in tree-based methods

#### **Open Source Impact:**
- Made high-performance gradient boosting accessible
- Enabled widespread adoption in industry and academia
- Inspired development of competing libraries (LightGBM, CatBoost)

### Useful Resources:
- Original XGBoost paper: https://arxiv.org/pdf/1603.02754
- XGBoost documentation: https://xgboost.readthedocs.io/
- Tianqi Chen's talks on XGBoost design principles
- "Elements of Statistical Learning" Chapter 10 on Boosting


***

## 2. 🔬 Key Technical Details

### Method

#### 1. **Gradient Tree Boosting Framework**
- Additive model: `ŷᵢ = Σₖ fₖ(xᵢ)` where each `fₖ` is a tree
- Objective function with regularization:
  ```
  L(φ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
  ```
- Second-order approximation using Taylor expansion
- Regularization term: `Ω(f) = γT + ½λ||w||²`

#### 2. **Sparsity-Aware Split Finding**
- **Problem**: Real data often has missing values or sparse features
- **Default direction**: Each tree node learns a default direction for missing values
- **Algorithm**: When feature value is missing, instance goes to default direction
- **Optimization**: Only visit non-missing entries during split finding
- **Impact**: Handles sparse data naturally without preprocessing

#### 3. **Weighted Quantile Sketch**
- **Problem**: Exact greedy algorithm too expensive for large datasets
- **Solution**: Approximate algorithm using quantile sketches
- **Key insight**: Use second-order gradient statistics as weights
- **Algorithm**: 
  - Propose candidate split points using weighted quantiles
  - Merge and prune sketches to maintain accuracy bounds
- **Guarantee**: ε-approximate quantile sketch with theoretical bounds

#### 4. **System Design Optimizations**

**Column Block for Parallel Learning:**
- Store data in compressed column blocks
- Enable parallel split finding across features
- Reduce memory access and improve cache efficiency

**Cache-aware Access:**
- Pre-sort data by feature values
- Use block structure to improve cache hit rates
- Minimize random memory access patterns

**Out-of-core Computation:**
- Divide data into multiple blocks stored on disk
- Use independent threads for block reading and gradient computation
- Overlap computation with I/O operations

### Key Algorithms and Techniques

#### 1. **Split Finding Algorithm**
```
Algorithm: Exact Greedy Algorithm for Split Finding
Input: I, instance set of current node
       d, feature dimension
1: gain ← 0
2: G ← Σᵢ∈I gᵢ, H ← Σᵢ∈I hᵢ
3: for k = 1 to d do
4:   GL ← 0, HL ← 0
5:   for j in sorted(I, by xⱼₖ) do
6:     GL ← GL + gⱼ, HL ← HL + hⱼ
7:     GR ← G - GL, HR ← H - HL
8:     score ← max(score, GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ))
9: return Split with max score
```

#### 2. **Approximate Algorithm with Sketches**
- Use weighted quantile sketch to find candidate split points
- Merge sketches from different machines in distributed setting
- Prune sketches to maintain memory bounds while preserving accuracy

#### 3. **Sparsity-Aware Split Finding**
```
Algorithm: Sparsity-aware Split Finding
1: for each feature k do
2:   // Only enumerate non-missing values
3:   for each j in sorted(Iₖ) do  // Iₖ = {i ∈ I : xᵢₖ ≠ missing}
4:     // Try both default directions
5:     Compute gain for left-default and right-default
6:   Choose best split and default direction
```

### System Architecture

#### **Block Structure:**
- **Compressed Sparse Column (CSC)**: Store features in column format
- **Block division**: Split columns into blocks for parallel processing  
- **Sorting**: Pre-sort each block by feature values
- **Compression**: Use compression to reduce memory footprint

#### **Cache Optimization:**
- **Access patterns**: Design algorithms to maximize cache locality
- **Prefetching**: Prefetch data blocks during computation
- **Memory hierarchy**: Optimize for different levels of memory hierarchy

#### **Out-of-core Learning:**
- **Block sharding**: Divide data into blocks that fit in memory
- **Asynchronous I/O**: Overlap disk I/O with computation
- **Gradient computation**: Compute gradients block by block

### Interesting Findings

#### **Performance Analysis:**
- **Speed**: 10x faster than existing solutions (R's gbm, scikit-learn)
- **Scalability**: Linear scaling with number of cores
- **Memory**: Efficient memory usage even for large datasets
- **Accuracy**: Maintains accuracy while being much faster

#### **Ablation Studies:**
- Sparsity-aware algorithm: 50x speedup on sparse datasets
- Weighted quantile sketch: Maintains accuracy with 10x speedup
- Cache-aware access: 2x speedup from better cache utilization
- Column block: Enables effective parallelization

#### **Real-world Impact:**
- **Kaggle dominance**: 17/29 winning solutions in 2015 used XGBoost
- **Industry adoption**: Used in production at major tech companies
- **Academic impact**: Thousands of citations and follow-up work

## 📚 References
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 785-794).
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

## Background Concepts: Gradient Boosting

### Historical Context
1. **Boosting Origins (1990s)**:
   - Theoretical foundations in PAC learning
   - AdaBoost: First practical boosting algorithm 
      - stump: weak learner, two leaf nodes for each condition
      - combine a lot of weak learner -> almost always stumps
      - each stump have different weights based on error rate 
      - each stump is made by taking the previous stump's error into account
      - often viewed as somewhat mysterious statistically
   - Focused on **binary classification** with exponential loss
   - SOTA method for tabluar data for more than a decade
  - late 90s - boosting as functional gradient descent: reweighting examples heuristically (like AdaBoost), boosting can be understood as minimizing a differentiable loss function by adding weak learners in a stage-wise manner.

2. **Gradient Boosting (2001)**:
   - Friedman generalized boosting to arbitrary differentiable loss functions
   - Key insight: Boosting as gradient descent in function space
      - taking small steps towards the right direction
   - Enabled regression and other tasks beyond classification

3. **Tree Boosting**:
   - Combined gradient boosting with decision trees
   - Trees as weak learners provide natural feature interactions
   - Became dominant approach for structured/tabular data
   - Essentially tree boosting is just gradient boosting + trees

### Gradient Boosting Algorithm

#### **Core Idea:**
Sequentially add models (weak learner) that correct errors of previous models:
```
F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
For m = 1 to M:
  1. Compute pseudo-residuals: rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₘ₋₁}
  2. Fit weak learner hₘ(x) to pseudo-residuals
  3. Find optimal step size: γₘ = argmin_γ Σᵢ L(yᵢ, Fₘ₋₁(xᵢ) + γhₘ(xᵢ))
  4. Update: Fₘ(x) = Fₘ₋₁(x) + γₘhₘ(x)
```

Each step: compute pseudo-residuals → fit a tree → add it to the ensemble.

#### **Why Trees Work Well:**
- **Feature interactions**: Trees naturally capture feature interactions
- **Non-parametric**: No assumptions about data distribution
- **Interpretability**: Tree structure provides interpretable rules
- **Robustness**: Handle mixed data types and missing values


## Background: Random Forest

### Context:
1. Decision Tree (60-80s):
    - CART (classification and regression tree) provide a simple & powerful way to partition data recursively
    - problem: overfit and have high variance
2. Bagging:
    - train multiple model on bootstrapped sample and average the output prediction
    - reduce variance but use the same base learner (CART)
    - But: if the dataset is large and the trees are deep, the bootstrap samples are often very similar.
        - → So the trees end up looking very similar too.
        - → That means their errors are correlated 
        - i.e. if one tree makes a mistake on some region, many others likely will too.
3. Random Subspace:
    - add randomness by selecting a subset of features for each tree to reduce correlation across tree
4. Rnadom Forests (2001):
    - combine bagging with random subspace at each split in the tree
    - reduce variance & keep the trees low-bias
    - robust, scalable, minimum tunning 

Impact:
- domainant algorithm in applied ML since 2001 until XGboost in 2014.
- considered a baseline model for tabluar data

### Method:
1. boostrap samping: random sample with replacement of size N (same size as dataset)
2. Growing tree with random feature splits
    - randomly select a subset of feature size m, m << d (all feature)
    - pick the best split (maximizes) among m using Gini impuristy, entropy, or variance reduction
3. tree depth
    - random forest grow fully expanded tree
    - individual tree might overfit, but averageing over can/may reduce variance
4. Ensemble aggregation:
    - classification: majority vote
    - regression: average prediction across all tree

Key Features of Random Forest:
1. Out-of-Bag (OOB) Error Estimate
2. Feature Importance
3. Robustness:
    - Handles high-dimensional data, missing values, and mixed data types.
    - Resistant to overfitting (because of averaging and randomness).

## Background: Random Forest vs. Gradient Boosting

Both are ensembling method.

| Aspect               | Random Forest (RF)           | Gradient Boosting (GB)              |
| -------------------- | ---------------------------- | ----------------------------------- |
| **Introduced**       | Breiman, 2001                | Friedman, 2001                      |
| **Family**           | Bagging (variance reduction) | Boosting (bias reduction)           |
| **Tree Building**    | Parallel, independent trees  | Sequential, each corrects previous  |
| **Bias–Variance**    | Low variance, modest bias    | Low bias, higher variance risk      |
| **Overfitting Risk** | Low                          | Higher (if not regularized)         |
| **Ease of Tuning**   | Few hyperparameters, simple  | Many hyperparameters, sensitive     |
| **Performance**      | Strong, stable baseline      | Often higher accuracy (with tuning) |
| **Speed**            | Parallelizable, fast         | Sequential, slower                  |

Random Forests = many independent trees, averaged → stable, variance reduction.

Gradient Boosting = sequential trees correcting errors → powerful, bias reduction but riskier.