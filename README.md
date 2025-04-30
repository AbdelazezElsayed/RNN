# Recurrent Neural Network with Backpropagation

This repository contains a Recurrent Neural Network (RNN) implemented in Python. It takes a sequence of 3 words ("I", "like", "to") and predicts the fourth word ("code") using a vocabulary of 4 words. The network has 4 input nodes (vocabulary size), 10 hidden neurons, and 4 output nodes, using tanh for hidden layers and softmax for outputs. Weights are randomly initialized with a small scale (0.01), and biases are set to zero. Built from scratch using NumPy, it demonstrates feedforward and backpropagation through time for training.

# Features





3-word input sequence, 1-word output prediction



Tanh (hidden) and softmax (output) activation functions



Random weights scaled by 0.01



Zero-initialized biases



Backpropagation through time to adjust weights and biases
