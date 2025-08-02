import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Input, Dense, Flatten, Conv2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def load_data():
    """Loading CIFAR-10 dataset."""
    return __loader__.load_data()

def model_configuration():
    """Configure the model parameters."""

    # Loading data for computing dataset size
    (input_train, _), (_, _) = load_data()
    
    # Generic config
    width, heigth, channels = 32, 32, 3
    batch_size = 64
    num_classes = 10
    validation_split = 0.1      # 10% of the training data will be used for validation
    verbose = 1
    n = 3  # Number of blocks in each stage
    init_fm_dim = 16
    shortcut_type = 'identity'  # 'conv' or 'identity' or 'projection'

    # epochs = 100

    # size of the dataset
    train_size = (1 - validation_split) * input_train.shape[0]
    val_size = validation_split * input_train.shape[0]

    # Number of steps per epoch: depending on the batch size
    max_iter = 64000

    # converting to int instead of tensor's
    steps_per_epoch = int(train_size // batch_size)
    validation_steps = int(val_size // batch_size)
    epochs = int(max_iter // steps_per_epoch)

    # Define loss function
    loss = tf.keraaas.losses.categorical_crossentropy(from_logits=True)

    # Learning rate config as per He et al. (2015)
    boundaries = [32000, 48000]
    values = [0.1, 0.01, 0.001]
    lr_schedule = PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )

    initializer = tf.keras.initializers.HeNormal()

    # Define Optimizers
    optimizer_momentum = 0.9  # Momentum for SGD optimizer
    optimizer_additional_metrics = ["accuracy"]
    optimizer_momentum = SGD(learning_rate=lr_schedule, momentum=optimizer_momentum)

    # Loading TensorBoard callback
    tensorboard = TensorBoard(
        log_dir = os.path.join(os.getcwd(), "logs"),
        histogram_freq=1,
        write_images=True,
    )

    # Saving model checkpoints after every epoch
    checkpoints = ModelCheckpoint(
        filepath = os.path.join(os.getcwd(), "checkpoints", "resnet_cifar10.h5"),
        save_freq = "epoch",
        save_best_only = True,
        monitor = "val_loss",
        mode = "min",
        verbose = 1
    )

    # Early stopping to prevent overfitting
    # If the validation loss does not improve for 3 epochs, training will stop
    EarlyStopping = EarlyStopping(
        monitor = "val_loss",
        patience = 3,
        verbose = 1,
        restore_best_weights = True
    )

    # Add callbacks to the list
    callbacks = [
        tensorboard,
        checkpoints,
        EarlyStopping
        ]
    
    # Create a dictionary to hold all configurations
    config = {
        "input_shape": (width, heigth, channels),
        "batch_size": batch_size,
        "num_classes": num_classes,
        "validation_split": validation_split,
        "verbose": verbose,
        "n": n,
        "init_fm_dim": init_fm_dim,
        "shortcut_type": shortcut_type,
        "train_size" : train_size,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "epochs": epochs,
        "loss": loss,
        "lr_schedule": lr_schedule,
        "optimizer_additional_metrics": optimizer_additional_metrics,
        "optimizer_momentum": optimizer_momentum,
        "initializer": initializer,
        "callbacks": callbacks
    }
    return config
