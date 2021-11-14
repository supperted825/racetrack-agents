from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Permute, LSTM, Reshape

# Various Neural Network Architectures are Defined Here

def DoubleConv(obs_shape):

    model = Sequential()

    model.add(Permute((3,2,1), input_shape=obs_shape))
    model.add(Conv2D(64,(3,3)))
    model.add(Conv2D(64,(3,3)))
    model.add(Flatten())
    model.add(Dense(256))

    return model


def NatureCnn(obs_shape):
    
    model = Sequential()
    
    model.add(Permute((3,2,1), input_shape=obs_shape))
    
    model.add(Conv2D(32, 8, strides=4, kernel_initializer="orthogonal"))
    model.add(Conv2D(64, 4, strides=2, kernel_initializer="orthogonal"))
    model.add(Conv2D(64, 3, strides=1, kernel_initializer="orthogonal"))
    
    model.add(Flatten())
    model.add(Dense(256))
    
    return model


def ConvLSTM(obs_shape):

    model = Sequential()

    model.add(Permute((3,2,1), input_shape=obs_shape))
    model.add(Conv2D(256,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(256,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))

    model.add(Reshape((256,-1)))

    model.add(LSTM(256, return_sequences=True))
    model.add(Flatten())


def MLP(obs_shape):

    model = Sequential()

    model.add(Permute((3,2,1), input_shape=obs_shape))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))

    return model


def Identity(obs_shape):
    
    model = Sequential()
    model.add(Permute((3,2,1), input_shape=obs_shape))
    model.add(Flatten())
    
    return model


# For Retrieval of Architectures

model_factory = {
    "DoubleConv": DoubleConv,
    "NatureCnn" : NatureCnn,
    "ConvLSTM"  : ConvLSTM,
    "MLP"       : MLP,
    "Identity"  : Identity
}

def get_model(opt):
    return model_factory[opt.arch](opt.obs_dim)