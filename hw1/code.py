import numpy as np
from collections import Counter

def data_loader():
    with open('train','r') as file:
        train_data = file.read().split('\n')[:-1]
    with open('test','r') as file:
        test_data = file.read().split('\n')[:-1]
    return train_data, test_data

def parser(datum):
    email_addr, label, words = datum.split(' ',2)
    words = words.split()
    word_dict = dict(zip([words[i] for i in range(0, len(words), 2)], [words[i+1] for i in range(0, len(words), 2)]))
    if label == 'spam':
        label = 1
    else:
        label = 0
    return label, word_dict

def data_preprocessing(train_data, test_data):
    y_train = np.zeros(len(train_data))
    y_test = np.zeros(len(test_data))
    x_train = []
    x_test = []
    for i, datum in enumerate(train_data):
        label, word_dict = parser(datum)
        y_train[i] = label
        x_train.append(word_dict)
    for i, datum in enumerate(test_data):
        label, word_dict = parser(datum)
        y_test[i] = label
        x_test.append(word_dict)
    return x_train, y_train, x_test, y_test

def compute_prior(y_train):
    ratio = Counter(y_train)
    return ratio[1]/len(y_train), ratio[0]/len(y_train)

train_data, test_data = data_loader()
x_train, y_train, x_test, y_test = data_preprocessing(train_data, test_data)
p_spam, p_ham = compute_prior(y_train)
print('Prior:')
print(p_spam, p_ham)


