from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Permute

# Various Neural Network Architectures are Defined Here

def DoubleConv256():

    model = Sequential()

    model.add(Permute((3,2,1), input_shape=(4, 128, 128)))
    model.add(Conv2D(256,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(256,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64))

    return model

# For Retrieval of Architectures

model_factory = {
    "DoubleConv256" : DoubleConv256
}

def get_model(arch):
    return model_factory[arch]()