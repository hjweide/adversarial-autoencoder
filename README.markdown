The relevant blog post is here: [http://hjweide.github.io/adversarial-autoencoders](http://hjweide.github.io/adversarial-autoencoders)

A Lasagne and Theano implementation of the paper [Adversarial
Autoencoders](http://arxiv.org/abs/1511.05644) by Alireza Makhzani, Jonathon
Shlens, Navdeep Jaitly, and Ian Goodfellow.  

Several design choices were made based on the discussion on
[/r/machinelearning](https://www.reddit.com/r/MachineLearning/comments/3ybj4d/151105644_adversarial_autoencod
ers/?).

To use this code:

1. Download the [MNIST data files](http://yann.lecun.com/exdb/mnist/).
2. Unzip and copy to the mnist directory.
3. Run ```python train.py``` to train a model, the weights will be saved to the ```weights``` directory.
4. Run ```python plot.py``` to generate the visualizations.
