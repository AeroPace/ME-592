#Variational Autoencoder Code
from keras.models import Model
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.backend import random_normal
import tensorflow
from tensorflow import exp
from tensorflow import shape

def build():
    input_ = Input(shape=(40, 40, 2), name='input_tensor')
    # Level 1
    conv1_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_) 
    conv1_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv1_1)
    #### Level 2
    conv2_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_2)
    drop2_1 = Dropout(0.1)(conv2_1)
    conv2_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(drop2_1)
    ######### Level 3
    conv3_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2_2)
    conv3_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv3_1)

    conv3_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv3_2)
    conv3_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv3_3)
    #### Level 2
    conv2_3 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv3_4)
    drop2_2 = Dropout(0.1)(conv2_3)
    conv2_4 = Conv2D(32, (3, 3), padding='same', activation='relu')(drop2_2)
    # Level 1
    conv1_3 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv2_4)
    conv1_4 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv1_3)
    output = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(conv1_4)
    # End of Decoder

    model = Model(input_, output)
    return model