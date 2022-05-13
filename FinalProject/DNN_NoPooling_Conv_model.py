from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import Dropout
from keras.layers import concatenate


def build():
    input_ = Input(shape=(40, 40, 2), name='input_tensor')

    # Level 1
    conv1_1 = Dense(16, activation='relu')(input_) 
    conv1_2 = Dense(16, activation='relu')(conv1_1)

    #### Level 2
    #pool2_1 = MaxPooling2D((2, 2), padding='same')(conv1_2) 
    conv2_1 = Dense(32, activation='relu')(conv1_2)
    drop2_1 = Dropout(0.1)(conv2_1)
    conv2_2 = Dense(32, activation='relu')(drop2_1)

    ######### Level 3
    #pool3_1 = MaxPooling2D((2, 2), padding='same')(conv2_2) 
    conv3_1 = Dense(64,  activation='relu')(conv2_2)
    conv3_2 = Dense(64,  activation='relu')(conv3_1)

    conv3_3 = Dense(64, activation='relu')(conv3_2)
    conv3_4 = Dense(64, activation='relu')(conv3_3)
    #up3_1 = UpSampling2D()(conv3_4)

    #### Level 2
    #concat2 = concatenate([conv2_2, up3_1], axis=3)
    conv2_3 = Dense(32,  activation='relu')(conv3_4)
    drop2_2 = Dropout(0.1)(conv2_3)
    conv2_4 = Dense(32,  activation='relu')(drop2_2)
    #up2_1 = UpSampling2D()(conv2_4)

    # Level 1
    #concat1 = concatenate([conv1_2, up2_1], axis=3)
    conv1_3 = Dense(16,  activation='relu')(conv2_4)
    conv1_4 = Dense(16,  activation='relu')(conv1_3)
    output = Dense(1, activation='sigmoid')(conv1_4)

    model = Model(input_, output)
    return model


