from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

N_DIGITS = 5

def get_mnist_data(n_digits=N_DIGITS):
    mnist = input_data.read_data_sets('.', one_hot=False)
    trainImg = mnist.train.images.reshape(-1, 28, 28)
    trainLabel = mnist.train.labels
    n_total = mnist.train.num_examples

    new_trainImg_list = []
    for i in range(n_total//n_digits):
        split_list = np.vsplit(trainImg[i*n_digits: (i+1)*n_digits], n_digits)
        split_list = [i.reshape(28, 28) for i in split_list]
        digits = np.hstack(split_list)
        new_trainImg_list.append(np.expand_dims(digits, 0))
    new_trainImg = np.vstack(new_trainImg_list)
    new_trainLabel = trainLabel.reshape(-1, n_digits)
    return new_trainImg, new_trainLabel.tolist()

if __name__ == '__main__':
    trainImg, trainLabel = get_mnist_data()
    plt.imshow(trainImg[0])
    print(trainLabel[0])
    plt.show()
