from os import walk, getcwd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Dense, Convolution2D,Dense,Dropout,Flatten
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot
from matplotlib import pyplot as plt

import h5py as h5py
from keras.models import load_model



if(__name__=="__main__"):
    mypath = "data/"
    txt_name_list = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        if filenames != '.DS_Store':       ##Ugh mac junk
            txt_name_list.extend(filenames)
            break
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    xtotal = []
    ytotal = []
    slice_train = int(80000/len(txt_name_list))  ###Setting value to be 80000 for the final dataset
    i = 0
    class_names = {0:"book", 1:"clock", 2:"hand", 3:"car", 4:"fish", 5:"laptop", 6:"cup"}
    order = ["book.npy","clock.npy","hand.npy","car.npy","fish.npy","laptop.npy","cup.npy"]

    seed = np.random.randint(5)
    for txt_name in order:
        print(txt_name)
        txt_path = mypath + txt_name
        x = np.load(txt_path)
        x = x.astype('float32') / 255.    ##scale images
        y = [i] * len(x)
        x = x.reshape(x.shape[0],28,28)
        np.random.seed(seed)
        np.random.shuffle(x)
        '''
        import matplotlib.pyplot as plt
        # print(class_names[y_train[555]])
        face1 = x[55]

        plt.imshow(face1)
        plt.show()
        exit()
        '''
        np.random.seed(seed)
        np.random.shuffle(y)
        x = x[:slice_train]
        y = y[:slice_train]
        if i != 0:
            xtotal = np.concatenate((x,xtotal), axis=0)
            ytotal = np.concatenate((y,ytotal), axis=0)
        else:
            xtotal = x
            ytotal = y
        i += 1
    x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train1 = np.zeros((y_train.size, y_train.max() + 1))
    y_train1[np.arange(y_train.size), y_train] = 1 #one hot encoding
    print(y_train1.shape)

    y_test1 = np.zeros((y_test.size, y_test.max() + 1))
    y_test1[np.arange(y_test.size), y_test] = 1 #one hot encoding
    print(y_train1.shape)




    model = load_model("digit_model.h5")
    # reading output

    op = model.predict(x_test)
    total = 0
    correct = 0
    for i in range(y_test.shape[0]):
        if(np.argmax(op[i]) == y_test[i]):
            correct = correct + 1
        total = total + 1
    print(correct,total)
    print("Test accuracy is:",str(correct/total))

    while True:
        id = int(input("Enter array testing index:"))
        if (id == "-1"):
            break
        else:
            a = x_test[id]
            a = a.reshape(28, 28)
            plt.imshow(a)
            plt.show()
            print(class_names[np.argmax(op[id])])

