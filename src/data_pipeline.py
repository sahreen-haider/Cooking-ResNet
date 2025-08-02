from tensorflow.keras.datasets import cifar10


class Loader:
    """Class to handle data loading and preprocessing."""

    @staticmethod
    def load_data():
        """Load CIFAR-10 dataset."""
        return cifar10.load_data()

    @staticmethod
    def preprocess_data(x, y):
        """Preprocess the data."""
        x = x.astype('float32') / 255.0  # Normalize the images
        y = tf.keras.utils.to_categorical(y, num_classes=10)  # One-hot encode labels
        return x, y