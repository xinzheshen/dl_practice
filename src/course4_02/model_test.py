import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import kt_utils

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def model(input_shape):
    """
    模型大纲
    :param input_shape:
    :return:
    """
    # 定义一个tensor的placeholder 维度为input_shape
    X_input = Input(input_shape)

    # 使用0填充：X_input 周围填充0
    X = ZeroPadding2D((3, 3))(X_input)

    # 对X使用CONV -> BN -> RELU块
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((2, 2), name="max_pool")(X)

    # 降维， 矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # 创建模型，创建一个模型实体，可以用它来训练 测试
    model = Model(inputs=X_input, outputs=X, name="HappyModel")

    return model


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = kt_utils.load_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    #创建一个模型实体
    happy_model = model(X_train.shape[1:])
    #编译模型
    happy_model.compile("adam","binary_crossentropy", metrics=['accuracy'])
    #训练模型
    #请注意，此操作会花费你大约6-10分钟。
    happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
    #评估模型
    preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
    print ("误差值 = " + str(preds[0]))
    print ("准确度 = " + str(preds[1]))
