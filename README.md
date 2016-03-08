[![Big surprise - memory is important to computers](http://i.imgur.com/YAPwro6.jpg)](https://www.flickr.com/photos/cheetleys/129757175/)

# Fully Differentiable Hierarchical Attentive Memory in TensorFlow

This repository contains a work-in-progress implementation of the architecture described in [Learning Efficient Algorithms with Hierarchical Attentive Memory](http://arxiv.org/abs/1602.03218) for TensorFlow.

## Hierarchical Attentive Memory Overview

Hierarchical Attentive Memory (HAM) allows for memory access in deep learning models in $O(\log n)$ complexity, substantially faster than the traditional neural attention mechanisms that require a full $O(n)$ traversal.
This is achieved by accessing memory in a tree like fashion, similar to hierarchical softmax.

The architecture can learn to sort $n$ numbers in $O(n \log n)$ and is able to generalize well to larger input sizes than what it was trained on.
Traditional LSTMs augmented with attention fail at both the sorting task and generalizing to larger inputs.

## Code Overview

The initial codebase implements the HAM architecture and uses the resulting model to select the smallest value from a given input.
Currently it is directly comparable to the Raw HAM module of Section 4.3 as it is not driven by an LSTM.
The plan is to extend the code to perform the sorting task as described in Section 4.2.

As opposed to the HAM architecture implemented in the paper, this is the fully differentiable version referred to as DHAM (see _Appendix A - Using soft attention_).
According to the paper, the DHAM architecture is slightly easier to train than the REINFORCE versio but does not generalize as well to larger memory sizes.

By default, `python ham_sort.py` will train the HAM architecture to select the smallest input from $n=8$ elements.
The weights are saved in `ham.weights` at the end of each epoch and will be re-used if training is restarted.
This allows for naive curriculum schedule training by training at smaller $n$, waiting for loss or accuracy to stabilize, and then increasing $n$ and restarting training.
