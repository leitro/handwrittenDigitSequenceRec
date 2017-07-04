# handwrittenDigitSequenceRec


## My software environment:

- Ubuntu 16.04 x64
- Python 3.5
- Tensorflow 1.1


## Structure:

It is a handwriting digits sequence recognition system. The datasets are digit sequences and labels which are based on MNIST datasets, and you can set the length of the training and testing sequence data by giving N_DIGITS_TR and N_DIGITS_TE the specific values in **handwrittenSequenceCTC.py**.

The CNN is used to extract high level features of the datasets, and LSTM is used to build the sequence to sequence model, and CTC is used as the loss function.

## Visualize Prediction Result Before CTC:

In the train() function of Class SeqLearn, test_flag can be set True to show the label error rate of test datasets, and visual_flag can be set True to visualize the prediction result before CTC. Both of the flags are set True by default.

- The 1st iteration, label_error_rate: 1.000
![1](https://user-images.githubusercontent.com/9562709/27832083-a53c60b0-60cd-11e7-8e7d-16a01c727b0e.jpg)

'''
decodedRes: 
[]
decoded without merge: 
[]
'''

- The 50th iteration, label_error_rate: 0.645
![50](https://user-images.githubusercontent.com/9562709/27832121-c13d0b5c-60cd-11e7-8ccd-7fd3047facd8.jpg)

'''
decodedRes: 
[[7 0 0 0]
 [9 0 0 0]
 [0 6 9 0]
 [9 7 4 0]]

decoded without merge: 
[[7 7 7 0 0 0 0]
 [9 9 9 0 0 0 0]
 [0 6 6 6 9 0 0]
 [9 7 7 7 4 0 0]]
'''

- The 400th iteration, label_error_rate: 0.048
![400](https://user-images.githubusercontent.com/9562709/27832129-cfd6ef52-60cd-11e7-9aa1-8d3561f81119.jpg)

'''
decodedRes: 
[[7 2 1 0 4]
 [1 4 9 8 9]
 [0 6 9 0 1]
 [5 9 7 3 4]]

decoded without merge: 
[[7 7 2 2 1 1 0 0 4 4 4 0]
 [1 1 1 4 9 8 9 9 0 0 0 0]
 [0 0 6 6 9 9 0 0 0 1 1 1]
 [5 9 7 7 7 3 3 4 4 0 0 0]]
'''

## Result:
After 400 epochs, the label error rate of test datasets has decreased to 0.048, the prediction accuracy has reached to 95.3%. 
