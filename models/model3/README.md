## Gradient-Based Learning with LeNet model

**What is it?**

Pattern recognition systems perform better when built relying more on automatic learning and less on hand-designed heuristics. Usually the system is divided into two partes. First a feature extractor, which transforms the input patterns so that yjey can be represented by low-dimensional vectors that can be easily compared and are relatively invariant to transformations and distortions of the input patterns that don't change their nature. Secondly, a classifier which is often general-purpose and trainable.

To train this system, gradient-based learning is often used. A loss function  which compares the desired output with the predictions of the model is minimized with procedures like gradient descent. This is possible in non-linear systems with several layers of processing thanks to algorithms like back-propagation used to calculate gradients efficiently by propagation from the output to the input. 


A real-world example for logistic regression could be as a predictor for wether a patient has a disease, like diabetes, or not, based on certain characteristics of the patient like age, body mass index, sex, results tests, etc.

**Strengths of the model**

Multi-layer networks trained with gradient descent are capable to learn high-dimensional and non-linear mappings from large collections of examples. This makes them suitable for image recognition tasks.

The LeNet model uses a specific architecture known as Convolutional Networks. This type of networks combines local receptive fields, shared weights and spatial/temporal sub-sampling to ensure a certain degree of invariance to shift, scale and distortion in the input. The local receptive fields allow the network to extract elementary visual features that are combined in following layers to detect higher-order features.

Convolutional networks are particularly well suited for recognizing or rejecting shapes with widely varying size, position, and orientation.

**Weaknesses of the model**

This model also has it has its trade-offs and weaknesses. First of all, neural networks tend to be more computationally expensive and require large amounts of data to train. 