# Core Components of a Transformer

## Inpute Representation
- Word or tokens are mapped into embeddigns (vectors).
- Positional encodings for the model to know the word order -> because attention has no notion of sequence

## Self-Attention
- Each token embedding is projected into 3 vectores:
    - Query (Q)
    - Key (K)
    - Value (V)
- Attention computes similiarity between queries and keys to decide which other word are important for a given token
$$
softmax(  Q \cdot K^T / \sqrt(d_k)  ) \cdot V
$$

Similarity Score:
$$
Q \cdot K^T
$$

Softmax: normalization for scores to turn into probability -> attention weights

. V: Multiply the wieghts by the values (V), translate the probability back to the actual token


## Multi-Head Attention
- Instead of one attention operation, run multiple in parallel (with different learned projections).
- Each "head" learns to capture a different type of relation (e.g., syntax, semantics).
- Outputs are concatenated and linearly transformed.

- Each head has its own projection matrices: 
$$
(W_h^Q, W_h^K, W_h^V)
$$


LLM training prevents all heads from converging to the same weights because of:
- Random initialization → different starting points.
- Different subspaces ( dk split across heads)
- Task gradients → redundancy doesn’t help loss, so complementary heads are favored.
- Regularization/noise → discourages collapse.


## Feedforward Network (FFN)
- After attention, each token goes through a small MLP (two linear layers with a nonlinearity).
- Adds depth and nonlinear reasoning capacity.

## Residual Connections + Layer Normalization
- Each sublayer (attention or FFN) has a skip connection and normalization.
- Helps stabilize training and enables very deep networks. 


## GeLU vs. ReLU
GeLU acts like a soft probabilistic gate
`GeLU(x)=x⋅Φ(x)`
Smoothly interpolates between 0 and x.

# Intuition Behind Layers
Think of multiple Transformer layers like a series of reasoning steps:

- Early layers: Capture local/simple patterns (word identity, neighboring relations).
- Middle layers: Build more abstract relations (syntax, phrase-level dependencies).
- Deeper layers: Capture high-level semantics and task-specific abstractions (long-range dependencies, world knowledge, discourse).

Each layer refines the representation of tokens, passing them upward like a processing pipeline.

# Transformer Architecture Variants
- Encoder-only (e.g., BERT): Good for understanding text.
- Decoder-only (e.g., GPT): Good for generating text.
- Encoder–decoder (e.g., original Transformer, T5): Good for translation and sequence-to-sequence tasks.


# Encoder-Only
Architecture: Keep the encoder stack, discard the decoder.
- Each token attends to all tokens bidirectionally (full context).
- Output: a contextual embedding per token.

Good for:
- Understanding tasks: classification, sentiment, NER, question answering (extractive).
- Because embeddings encode global context.

# Decoder-Only
Architecture: Keep the decoder stack, discard the encoder.
- Each token attends only to previous tokens (causal, left-to-right masking).
- Output: probability distribution over the next token.