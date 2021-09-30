import os
import tensorflow as tf

from keras.callbacks import TensorBoard
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

    return model

# For Retrieval of Architectures

model_factory = {
    "DoubleConv256" : DoubleConv256
}

def get_model(arch):
    return model_factory[arch]()


"""Custom Tensorboard, modified from PythonProgramming.net by Mohammed AL-Ma'amari for RL Models."""

class ModifiedTensorBoard(TensorBoard):
    
    # Override to set initial step and writer (one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Override to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Override to save logs with current step number
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Override to train for one batch only, no need to save at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Override to keep writer open
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
