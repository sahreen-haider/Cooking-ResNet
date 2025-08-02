import numpy as np

class Augmentation:
    def __init__(self) -> None:
        pass

    def random_crop(img: np.matrix, random_crop_size: tuple) -> np.matrix:
        """Randomly crops the image to the specified size."""
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx +1)
        y = np.random.randint(0, height - dy +1)
        return img[y:(y + dy), x:(x + dx), :]
