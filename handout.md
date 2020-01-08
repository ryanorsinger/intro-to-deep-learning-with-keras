# Intro to Deep Learning

## Summary
This workshop will introduce the conceptual basis of 


## What is Deep Learning?

### Why and when to use ANNs vs. Classical ML tools

### Why and when to use Classical ML over ANNs

## Proper Feeding of Neural Nets

- Vector data is a 2-dimensionsal tensor (1 dimension for the samples, **features**)
- Time series data or sequence data are 3D tensors of shape (samples, time-steps,
  **features**)
- Images—4D tensors of shape(samples,height,width,channels)or(samples,
  channels, height, width)
- Video —5D tensors of shape (samples, frames, height, width, channels) or
  (samples, frames, channels, height, width)

 <img alt="4d data" src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/02fig04.jpg" width="230">


## What do Multi-Layer Perceptrons (known as MLPs or ANNs) really do? 

This figure shows the original data and a new representation. We obtain new representations of the data by changing our axes to fit the story the data tells. Here we get a new representation of the data in such a way that we can describe data with a simple rule:

> “Black points are such that x > 0,” or “White points are such that x < 0.”

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig04.jpg" >



Geometric transformations in high dimensional space. "Uncrumpling paper balls is what machine learning is about: finding neat representations for complex, highly folded data manifolds. Each layer in a deep network applies a transformation that disentangles the data a little" - Chollet


## Anatomy of a Neural Network (in 3 figures)


### How does the learning actually happen?


## Questions

What's the relationsihp betweeen artificial neural networks and real neurons?

What are neural networks and deep learning?

How are artificial neural networks (ANNs) different than other machine learning algorithms?

When should we use ANNs vs. classical machine learning?
- Tremendous volumes of unstructured data? An ANN may be the right tool

The central problem in deep learning is to meaningully transform data to learn useful representations of the input data so those representations get us closer to the desired output.x

What's a representation? A different way to look at data. For example, the RGB representation of "red" is [1, 0, 0] not the string "red".

## Key Concepts and Vocabulary
Tensor
Perceptron
Multi-Layer Perceptron
Feed forward
Backpropogation
Activation Function
Loss
Optimizer

## Keras Workflow
0. Prep and load your data. You'll nemeaning it's already a tensor or you've converted your data to be a tensor)
1. Create the model
2. Add layer(s) with .add
3. Compile the model with .compile (to configure the learning parameters)
4. Fit the model with .fit
5. Evaluate model performance with .evaluate
6. Produce predictions on new data with .predict
7. Decode the prediction data if necessary back to its original representation (numbers, text, images)



