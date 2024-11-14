# MNIST-Neural-Network

## Overview
### What is MNIST?
The MNIST training/testing sets include 60,000 / 10,000 handritten integers from 0 - 9. Each value is broken down into a 28x28 pixel format, and laied out into a 1-D array with a length of 784 pixels.

### How can we read it?
Using Java with no built in matrix or neural network libraries, I have built a neural network that is trained with the 60,000 MNIST dataset based on epoch size, mini batch size, and many other parameters. 
At a minimum, the network is 3 layers: Input -> Hidden -> Output, but the number of hidden layers can be adjusted to any amount. The calculus and matrix math behind activation functions and backpropagation are fully implemented. The program is ran soley in the terminal, but upon running, it offers several features for the user to play with including: 

1) Loading a pretrained network
2) Train the network on the training set
3) Save the trained network to a file
4) See statistics (correct/incorrect) on training data
5) See statistics (correct/incorrect) on testing data
6) Display each number and see the actual value vs the predicted.
7) View only misclassified numbers (incorrect)
