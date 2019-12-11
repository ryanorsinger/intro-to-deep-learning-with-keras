## Vocabulary
Neuron
Perceptron - a single neuron with n amount of binary inputs that computes a weighted sum of its in puts and "fires" if that weighted sum is 0 or greater.








Questions:
What is a neuron?
How do DL neurons differ?
What isa perceptron?

What is an Artificial Neural Network?
What is a multilayer perceptron?

How do we build, train, evaluate, and run neural networks?

ANNs go back to 1943 w/ the McCullouch and Pitts paper "Logical Calculus of Ideas Immanent in Nervous Activity". They provided a computational model of how 

Why ANNs?
- With enough data, ANNs can outperform other ML techniques
- Virtuous cycle of funding and progress



Neural Network:
lots of simple units that do simple things but when connected, they can accomplish impressive feats of computation

Logical Computations with Neurons
- Identity function, And, Or, Not
- 

Think of a neuron as a function that takes in one or more inputs, performs a calculation, and then fires (sends output) if the result of the calculation exceeds a threshhold or not.


An artificial neuron:
- one or more binary inputs and one binary output
- activates its output when more than a certain number of its inputs are active (some threshhold)
- The identity function C = A. If neuron A is activated, then neuron C gets activated as well (since it receives two input signals from neuron A); but if neuron A is off, then neuron C is off as well

A perceptron is one of the simplest ANN architectures. Perceptrons have numeric inputs and outputs instead of booleans and each input connection is associatied with a weight. 
- The perceptron works by computing a weighted sum of the inputs (z = x'w)
- We take a list of features and a list of weights and product the inner product
- Then applies a "step function" to that sum and outputs the result.

### Common Types of Step functions
heaviside(z) = {0 if z < 0, 1 if z >= 0}
sng(z) = -1 if z < 0, 0 if z = 0, +1 if z > 0

usually, heaviside is the default step function, but we can specify others.

The perceptron is simply distinguishing between the half-spaces separated by the hyperplane of points x for which:
dot(weights, x) + bias == 0

With the right weights, a single perceptron can solve:
- Create an AND gate/boundary
- Create an OR gate/boundary
- NOT boundary

Like real neurons, artificial neurons start getting more interesting when you start connecting them together.

## Feed Forward Neural Networks vs. Perceptrons
- multiple layers of multiple perceptrons

- each layer is connected to the next
-
 - each neuron we’ll sum up the products of its inputs and its weights, then determine if those pass a threshhold


### Neural networks take vectors as inputs and produce vectors as outputs
- To use a NN, our first challenge is to come up with a way to recast the programming problem as a vector problem.

If we're building an encoder for string input that represents a red, yellow, green light on a stoplight. The encoding would be the following. That's because the input to the NN needs to be represented as a vector.

def stoplight_encode(x):
    if x == "green":
        return [1, 0, 0]
    elif x == "yellow":
        return [0, 1, 0]
    else:
        return [0, 0, 1]
