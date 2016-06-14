# DeepDoomLearning

This is an implementation of Deep Reinforcement Learning as proposed from DeepMind 
(see: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

It is based on the ViZDoom (https://github.com/Marqt/ViZDoom) learning environment. The deep convolutional network is implemented in tensorflow (https://www.tensorflow.org/)

This proof of concept implementation, can serve as a starting point for further investigation. The implemented deep network is able to learn the basic scenario from ViZDoom within 1 hour on a GeForce GTX 970 + Core i7 4770.

For further learning more complex scenarios an addional layer should be added to the network.

Caution: This is work in progress and needs some refactoring. For this to work you need ViZDoom and Tensorflow.
