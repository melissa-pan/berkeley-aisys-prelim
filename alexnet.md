# ImageNet Classification with Deep Convolutional Neural Networks

## 📋 Basic Information
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Year**: 2012
- **Venue**: NeurIPS (NIPS)
- **Link**: [PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)



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
  - First successful application of deep CNNs to large-scale image classification
  - Demonstrated the power of end-to-end learning for computer vision
  - Popularized ReLU activation functions
  - Showed effectiveness of dropout regularization

- **How did this paper change the field after its release?**


### [Optional] Background & History
- **Useful background knowledge:**
  - Convolutional Neural Networks (LeCun et al., 1998)
  - ImageNet dataset and competition
  - GPU computing for deep learning
  - Backpropagation and gradient descent

#### **Pre-history and context:**
  - Computer vision was dominated by hand-engineered features
  - Neural networks were considered impractical for large-scale problems
  - Limited computational resources prevented training deep networks

I found this talk to be very insightful [https://youtu.be/Z7naK1uq1F8?feature=shared](https://youtu.be/Z7naK1uq1F8?feature=shared) Feifei's talk on the human story behind imagenet, & why she believed imagenet is needed.
- Vision is an important human capability, 40% of the human brain neuron is related to vision.
- Before imagenet, classification are done on 20+ labels. But human can recongize 30k categories by age six. Fitting model on 20 labels is foundamentally off from real human capability.
- Thus we need imagenet (millions of object), but, scaling human annotation is challenging (so true). 
- Imagenet became the benchmark (with competitions) in visions, which is what AlexNet participated in.



***

## 2. 🔬 Key Technical Details

### Deep Method Understanding
- **How does the method work in detail?**


- **Key algorithms and techniques:**

### Case Studies & Examples
- **Specific examples and implementations:**


### Technical Insights
- **Deep technical understanding and nuances:**

## ❓ Personal Questions to Answer
- **What did I learn from this paper?**
  - 

- **What questions do I still have?**
  - 

- **How does this relate to other papers I've read?**
  - 

- **What would I do differently if I were to implement this?**
  - 


## 📚 References
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.


## Background Concepts: Backprop
1. Historical and Contextual Background
- Early Neural Networks (1940s–1960s):
In the 1950s–1960s, Rosenblatt introduced the perceptron, an early learning algorithm. However, perceptrons were limited: they could only solve linearly separable problems (like AND/OR) but not nonlinear ones (like XOR). -> backprop doesn't work here.
- Backprop doesn't work here is due to architectural limitation of perceptron (single layer network)
   1. Non-differentiable Activation: no meaningful gradient (derivative is zero almost everywhere). Backprop requires smooth, differentiable activations (like sigmoid, tanh, ReLU) to propagate error signals.
   2. Too Simple Architecture: Even if you replaced the step function with a smooth activation, a single-layer perceptron is still limited to linear separation. Backprop wouldn’t “fix” that — it would just converge to the best linear boundary possible.

2. The Problem of Training Multi-layer Networks:
Researchers realized that more complex networks (multi-layer perceptrons, MLPs) could, in principle, represent more sophisticated functions. The challenge was how to efficiently train them. Early attempts relied on brute-force or biologically inspired methods, but none scaled well.
   - Brute-force “finite difference” methods: Perturb each weight slightly, measure the effect on the loss, and approximate the gradient numerically. This requires two forward passes per weight. For networks with thousands (today, billions) of weights, this is computationally impossible.
   - Random search / evolutionary methods: Adjust weights randomly and keep changes if performance improves. Works for tiny networks but becomes exponentially slower as networks grow.
   - Layer-by-layer heuristics: Train one layer at a time (e.g., unsupervised methods for hidden layers, then train the output layer). This avoids dealing with full gradients, but doesn’t reliably optimize the whole network.
   - Biologically inspired “Hebbian learning”: “Neurons that fire together wire together” — local update rules. Intuitively plausible, but not aligned with minimizing a global error signal in deep, multi-layer systems.

3. Backpropagation Emerges (1970s–1980s):
The algorithm we now call backpropagation was independently discovered multiple times. The key milestones were:
   - Paul Werbos (1974) first described using the chain rule for training multi-layer networks in his PhD thesis.
   - David Rumelhart, Geoffrey Hinton, and Ronald Williams popularized backpropagation in their influential 1986 paper, showing it could effectively train deep neural networks.

source: [link](https://chatgpt.com/s/t_68b39ccaa58c8191b72a523beb1d0baf)

### Backprop
application of the chain rule of calculus to compute gradients efficiently in multi-layer neural networks.

1. Forward Pass:
  - Input data flows through the network layer by layer.
  - Each neuron applies a weighted sum and an activation function.
  - At the output layer, the network produces predictions, and the loss function compares these with the true labels.

2. Backward Pass (Gradient Computation):
  - Start at the output layer: compute the gradient of the loss with respect to the output (error signal).
  - Propagate this gradient backward using the chain rule:
     - For each layer, compute the derivative of the loss with respect to the layer’s weights and activations.
     - Pass the error signal back to earlier layers, adjusting for each layer’s parameters.
  - This creates a recursive relationship: the gradient at one layer depends on the gradient from the layer above it.

3. Weight Update:
  - Use gradient descent (or a variant like Adam) to update each weight:
  w := w - n ( ∂W / ∂L)



### Impact on AI:
- Backpropagation revived interest in neural networks after a period of stagnation ("AI winter") and became the cornerstone of modern deep learning. It enabled large-scale training of models in fields like computer vision, speech recognition, and later, natural language processing.
- Backpropagation is still the core algorithm for training deep learning models, though often combined with optimizers (Adam, RMSProp) and regularization techniques (dropout, batch norm).
- Some research explores alternatives (like Hebbian learning, evolutionary methods, or biologically plausible algorithms), but backpropagation remains dominant due to its efficiency and effectiveness.


### Useful Resource for Backprop:
- CMU deep learning system lectures: https://dlsyscourse.org/lectures/
   - Manual Neural Networks / Backprop: https://youtu.be/OyrqSYJs7NQ?feature=shared
