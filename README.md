# Modern Hopfield Networks on the CIFAR10 dataset
**2025 Spring CSCI1470 Deep Learning Final Project**

**Authors: [Haosheng Wang](https://github.com/Wonder947), Edrick Guerrero, [Alfonso Gordon Cabello de los Cobos](https://github.com/AlfonsoR-GordonCC)**
## Summary
The idea of this project is to implement Modern Hopfield Networks on the CIFAR10 dataset for classification of the different elements inside the set. This will be done by using TensorFlow as a way of providing another alternative than PyTorch, wich is theone used originally in [Hopfield Networks is All You Need](https://github.com/ml-jku/hopfield-layers) by [Institute for Machine Learning, Johannes Kepler University Linz](https://github.com/ml-jku)
.

## Reflection (2025 04 26)
### Introduction
Memory involves the efficient storage and retrieval of information, and it comes in various forms—short-term, long-term, sensory, procedural, among others. Hopfield networks, also known as associative memories, are a class of recurrent neural networks (RNNs) designed to function as content-addressable memory systems. A defining characteristic of these networks is their ability to reconstruct entire patterns from partial or noisy inputs.

The original Hopfield network, introduced by John J. Hopfield in 1982, was based on binary feature representations and binary activation functions. Since then, significant advancements have been made. Modern Hopfield networks generalize the original model to continuous states, dramatically increasing storage capacity and stability. These developments, particularly in networks with continuous dynamics and large memory capacity, have been explored in a series of works since 2016.

In this project, we focus on replicating and analyzing the 2020 paper "Hopfield Networks is All You Need", which presents a modern formulation of Hopfield networks that bridges them with attention mechanisms commonly used in deep learning nowadays.

### Challenges
- What has been the hardest part of the project you’ve encountered so far?
- A: the implmentation is relatively easy, but to understand the underlying math proof about questions such as why the storage capacity would be exponential w.r.t. state dims, why the new energy function and update rule (almost) guarantees converge to local min or saddle points after one update is hard.

### Insights
- Are there any concrete results you can show at this point?
- A: We have tested the performance of classical hopfield network on remembering some black-white images and retrieve with half of the image masked. We have preprocessing functions required to add noise/mask to the images.

- How is your model performing compared with expectations?
- our model is being implemented at the time...

### Plan
- Are you on track with your project?
- yes!

- What do you need to dedicate more time to?
- understanding the math, try some improvements, and discuss about the criteria for measuring the quality of masking/noised images, the quality of retrieved images.

- What are you thinking of changing, if anything?
- currently, we are improving over the original notebook by doing experiments to find some more proper way to evaluate the performance of the modern hopfield network. Instead of just adding more images to remember, we are going to evaluate the difficulty of remembering different images systematically.


## Dependencies
Runing the script/notebook requires of the following libraries and versions:

- python = 3.11.9
- tensorflow == 2.15
- matplotlib
- numpy

