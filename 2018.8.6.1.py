
# -*- coding: utf-8 -*-  

'''Trains a simple deep NN on the MNIST dataset.



Gets to 98.40% test accuracy after 20 epochs

(there is *a lot* of margin for parameter tuning).

2 seconds per epoch on a K520 GPU.

'''

 

from __future__ import print_function

 

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

 

batch_size = 128

num_classes = 10

epochs = 20



# the data, shuffled and split between train and test sets 

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

 

import numpy as np

path='./mnist.npz'

f = np.load(path)

x_train, y_train = f['x_train'], f['y_train']

x_test, y_test = f['x_test'], f['y_test']

f.close()

 

x_train = x_train.reshape(60000, 784).astype('float32')

x_test = x_test.reshape(10000, 784).astype('float32')

x_train /= 255

x_test /= 255

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

 

# convert class vectors to binary class matrices

# label为0~9共10个类别，keras要求格式为binary class matrices

 

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

 

# add by hcq-20171106

# Dense of keras is full-connection.

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))


##Dense就是常用的全连接层，
##所实现的运算是output = activation(dot(input, kernel)+bias)。

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

 
##softmax 
##这是归一化的多分类，可以把K维实数域压缩到（0，1）的值域中，并且使得K个数值和为1。
##
##sigmoid 
##这时归一化的二元分类，可以把K维实数域压缩到近似为0，1二值上。
##
##relu 
##这也是常用的激活函数，它可以把K维实数域映射到[0,inf)区间。
##
##tanh 
##这时三角双曲正切函数，它可以把K维实数域映射到(-1,1)区间。
model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])


##优化器optimizer：该参数可指定为已预定义的优化器名，如rmsprop、adagrad，
##或一个Optimizer类的对象，详情见optimizers

##损失函数loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，
##如categorical_crossentropy、mse，也可以为一个损失函数。详情见losses

##指标列表metrics：对分类问题，我们一般将该列表设置为metrics=[‘accuracy’]。
##指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.
##指标函数应该返回单个张量,
##或一个完成metric_name - > metric_value映射的字典.请参考性能评估

history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=2,

                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
