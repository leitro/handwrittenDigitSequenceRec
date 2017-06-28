# handwrittenDigitSequenceRec

------------

## My software environment:

- Ubuntu 16.04 x64
- Python 3.5
- Tensorflow 1.1

----------

## Structure:

It is a simplified version of handwriting sequence recognition. The datasets are digit sequences and labels which are based on MNIST datasets, and you can set the length of the sequence using get_mnist_data() function in **mnist.py**. In the model, LSTM and CTC are used.

-------------

## Visualize demo:

The **showCTC.py** file is a demo for visualizing the prediction:

- The 1st iteration, batch_cost: 12.728
![](https://user-images.githubusercontent.com/9562709/27639548-d399474a-5c16-11e7-8d5d-9a0a0af39c90.png)

- The 50th iteration, batch_cost: 0.398
![](https://user-images.githubusercontent.com/9562709/27639572-e5821b4e-5c16-11e7-8d97-992812675d26.png)

- The 100th iteration, batch_cost: 0.118
![](https://user-images.githubusercontent.com/9562709/27639592-f7b39b44-5c16-11e7-8698-7d026ea84f70.png)

--------------

## TroubleShooting:

1. The shape of prediction is (time_steps, batch_size, num_classes+1), and in the demo above, only first 5 time steps are visualized. If we visualize all the time steps, it should be:

![](https://user-images.githubusercontent.com/9562709/27640134-ae42b830-5c18-11e7-99cf-15364d15697e.png)

We can see that the weights in the following time steps after the first 5 ones are meaningless, so it is not working as what we want it to be:

![](https://user-images.githubusercontent.com/9562709/27640862-9a56d0b6-5c1a-11e7-9fec-b968d81e5a0a.png)

So maybe our expected process above is done inside the CTC loss function, I need to dive into the tensorflow codes to find it out.

2. During the training using the main file **handwrittenSequenceCTC.py**, the cost keeps stable and never decreases, and sometimes it occurs "No valid path found.". I tried to use a low learning rate and high momentum, but the problem is still there. But in the **showCTC.py**, when I only focus on 1 example, the model works and the cost decreases dramatically. Until now, I cannot think of a way to fix it, so any advice? Thanks!
