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
    seed = np.random.randint(5) #for random: 1, 10e6
    for txt_name in order:
        print(txt_name)
        txt_path = mypath + txt_name
        x = np.load(txt_path)
        x = x.astype('float32') / 255.    ##scale images
        y = [i] * len(x)
        x = x.reshape(x.shape[0],28,28)
        print(x.shape)
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

    #displaying the image
    # a=x_train[3]
    # print(class_names[y_train[3]])
    # print(a.shape)
    # a = a.reshape(28,28)
    # print(a)
    # plt.imshow(a)
    # plt.show()
    #
    # exit()
    # x_test = np.expand_dims(x_test, axis=0)
    model=Sequential()
    model.add(Convolution2D(32,3,3,activation='relu',input_shape=(28,28,1)))
    # model.add(Convolution2D(32,3,3,activation='relu'))
    # model.add(Convolution2D(32,3,3,activation='relu'))
    # model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(Dropout(0.77))
    model.add(Flatten())
    model.add(Dropout(0.45))
    model.add(Dense(7,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc','mse', 'mae'])

    history = model.fit(x_train,y_train1,batch_size=64,nb_epoch=20,verbose=1,validation_split=0.1)

    outfile = open("op","wb")
    pickle.dump(history.history['mean_absolute_error'],outfile)
    pickle.dump(history.history['mean_squared_error'],outfile)
    pickle.dump(history.history['acc'],outfile)
    outfile.close()

    model.save('digit_model.h5')
    '''
    pyplot.plot(history.history['mean_absolute_error'])
    pyplot.plot(history.history['mean_squared_error'])
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Mean Absolute Error')
    # pyplot.show()
    pyplot.clf()#clears the entire current figure
    # pyplot.plot(history.history['acc'])
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Accuracy')
    '''
    # pyplot.show()

