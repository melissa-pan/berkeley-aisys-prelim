# Efficient Memory Management for Large Language Model Serving with PagedAttention

## 📋 Basic Information
- **Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Year**: 2023
- **Venue**: SOSP (ACM Symposium on Operating Systems Principles)
- **Link**: [PDF](https://arxiv.org/abs/2309.06180)



## 1. 📖 Paper Understanding

### The Problem

#### What problem does this paper solve?

- **Memory inefficiency in LLM serving**: High throughput serving of large language models requires batching many requests simultaneously, but existing systems struggle with inefficient Key-Value (KV) cache memory management
- **Dynamic memory allocation challenges**: KV cache memory for each request is huge and grows/shrinks dynamically during generation, leading to significant memory waste
- **Fragmentation and duplication**: Memory is wasted through fragmentation and redundant duplication of KV cache data, severely limiting the achievable batch size

#### Prior art and why they didn't work well:
- **FasterTransformer**: Static memory allocation leads to significant memory waste when sequences are shorter than the maximum length
- **Orca**: Better memory management than FasterTransformer but still suffers from fragmentation and cannot efficiently share KV cache between requests
- **Existing systems**: Lack of systematic approach to manage the dynamic, variable-length KV cache memory efficiently


### The Key Idea

**PagedAttention: OS-inspired memory management for LLM serving**

- **Virtual memory analogy**: Apply classical virtual memory and paging techniques from operating systems to manage KV cache memory
- **Block-based allocation**: Divide KV cache into fixed-size blocks that can be allocated dynamically and non-contiguously
- **Flexible sharing**: Enable efficient sharing of KV cache blocks within and across requests (e.g., for parallel sampling, prefix sharing)
- **Near-zero waste**: Achieve optimal memory utilization by eliminating fragmentation and enabling precise memory allocation

### The Challenge

#### What are the main challenges in solving this problem?

- **Dynamic sequence lengths**: LLM generation produces sequences of unpredictable and variable lengths, making static allocation inefficient
- **Memory fragmentation**: Traditional allocation methods lead to external fragmentation when sequences have different lengths
- **KV cache sharing complexity**: Efficiently sharing identical prefixes or handling beam search requires sophisticated memory management
- **Performance overhead**: Memory management system must be fast enough to not become a bottleneck in high-throughput serving
- **Attention computation adaptation**: Need to modify attention mechanisms to work with non-contiguous memory blocks

### The Method

#### Brief overview (detailed analysis in Section 2):
- **Paged memory organization**: Organize KV cache into fixed-size blocks (pages) that can be allocated dynamically
- **Indirection layer**: Use block tables to map logical sequence positions to physical memory blocks
- **Modified attention kernel**: Adapt attention computation to work with paged memory layout
- **Copy-on-write sharing**: Enable efficient sharing of KV cache blocks until modification is needed
- **Dynamic block allocation**: Allocate and deallocate blocks as sequences grow and complete

### Pros & Cons

#### Strengths:
- **Near-zero memory waste**: Eliminates fragmentation and enables precise memory allocation matching actual sequence lengths
- **Flexible KV cache sharing**: Efficient sharing within requests (beam search) and across requests (common prefixes)  
- **High throughput**: 2-4× improvement in serving throughput compared to state-of-the-art systems
- **Scalability**: Better performance gains with longer sequences, larger models, and more complex decoding
- **General applicability**: Works with various LLM architectures and decoding algorithms

#### Weaknesses/Limitations:
- **Implementation complexity**: Requires significant changes to attention kernels and memory management systems
- **Block size tuning**: Fixed block size requires careful tuning to balance memory efficiency and computational overhead
- **Limited to autoregressive models**: Primarily designed for decoder-only models with KV cache requirements
- **Hardware dependency**: Performance benefits depend on efficient implementation of paged attention kernels

### Impact & Contributions

#### Key contributions to the field:
- **Novel memory management paradigm**: First to systematically apply OS paging techniques to LLM serving
- **Practical serving system**: vLLM implementation demonstrates real-world applicability and performance gains
- **Attention algorithm innovation**: PagedAttention algorithm enables efficient computation with non-contiguous memory
- **Memory sharing mechanisms**: Introduced sophisticated KV cache sharing strategies for both intra and inter-request scenarios

#### How did this paper change the field after its release?
- **Industry adoption**: vLLM became widely adopted in production LLM serving systems
- **Serving optimization focus**: Shifted attention from training optimizations to inference and serving efficiency
- **Memory management research**: Inspired further research into memory-efficient LLM serving techniques
- **Open-source impact**: vLLM's open-source release democratized high-performance LLM serving



## 2. 🔬 Key Technical Details

### Method

**PagedAttention Algorithm:**
- **Block organization**: KV cache divided into fixed-size blocks, each storing attention keys and values for a fixed number of tokens
- **Block table mapping**: Each sequence maintains a block table that maps logical block indices to physical block addresses
- **Attention computation**: Modified attention kernels that can fetch key-value pairs from non-contiguous physical blocks
- **Dynamic allocation**: Blocks allocated on-demand as sequences grow, deallocated when sequences complete

#### Key algorithms and techniques:

1. **Paged Memory Management:**
- Fixed-size blocks (typically 16-256 tokens per block) store KV cache data
- Block tables provide indirection from logical sequence positions to physical memory
- Free block pool managed by centralized allocator
- Copy-on-write mechanism for efficient sharing before modification

2. **PagedAttention Kernel:**
- Modified attention computation to handle non-contiguous KV cache storage
- Efficient memory access patterns despite fragmentation
- Maintains computational efficiency comparable to contiguous memory layouts
- Supports various attention variants (multi-head, grouped-query attention)

3. **KV Cache Sharing Mechanisms:**
- **Parallel sampling**: Share prefixes across multiple samples from same prompt
- **Beam search**: Efficiently manage branching and merging of search paths  
- **Prefix sharing**: Share common prefixes across different requests
- **Copy-on-write**: Lazy copying ensures sharing until actual divergence occurs

4. **Memory Scheduling and Allocation:**
- Request scheduling based on available memory blocks
- Preemption and swapping for memory-constrained scenarios
- Block-level garbage collection and reallocation
- Memory pool management across multiple GPUs

### Small block size vs. large block size

Small:
- Pro: less memory waste
- Pro: faster recomputation if needed
- con: larger metadata overhead (bigger block table)
- con: not able to fully utilize GPU parallelism if too small
- con: more scattered memory accesses -> physical location can be different
- con: more memory op in allocation and freeing the block

Large:
- Pro: better memory locality
- Pro: less metadata
- Pro: less memory ops
- Con: internal fragmentation
- Con: bad for reuse (internal fragmentation)

### Recomputation vs. Swapping to CPU memory
if block size small then recomputation can be quick, we recompute all kv in one go

if block size big then swapping is better


all or nothing for eviction policy

### Useful Resources:
- Talk at SOSP: https://www.youtube.com/watch?v=UdNocRPQS3Y
- [vLLM Documentation](https://docs.vllm.ai/) - Implementation details and usage