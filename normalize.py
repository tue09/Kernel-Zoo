import numpy as np

class Normalize:
    def __init__(self, normalize_method):
        self.set_normalize(normalize_method)

    def set_normalize(self, normalize_method):
        if normalize_method == "None":
            self._normalize_method = self.identity
        elif normalize_method == "min-max":
            self._normalize_method = self.min_max_norm
        elif normalize_method == "sigmoid":
            self._normalize_method = self.sigmoid_norm
        elif normalize_method == "clipping":
            self._normalize_method = self.clipping
        elif normalize_method == "mean":
            self._normalize_method = self.mean_norm
        elif normalize_method == "std":
            self._normalize_method = self.std_norm
        elif normalize_method == "L1":
            self._normalize_method = self.L1_norm
        elif normalize_method == "L2":
            self._normalize_method = self.L2_norm
        else:
            print("Invalid normalization method!")

    def identity(self, image):
        return image

    def min_max_norm(self, image):
        image_normalized = (image - image.min()) / (image.max() - image.min() + 0.001) * 255
        return image_normalized

    def sigmoid_norm(self, image):
        image_normalized = (1 / (1 + np.exp(-image))) * 255
        return image_normalized

    def clipping(self, image):
        image_normalized = np.clip(image, 0, 255)
        return image_normalized

    def mean_norm(self, image):
        image_normalized = image - np.mean(image, axis=(0, 1), keepdims=True)
        return image_normalized

    def std_norm(self, image):
        image_normalized = (image - np.mean(image, axis=(0, 1), keepdims=True)) / np.std(image, axis=(0, 1), keepdims=True)
        return image_normalized

    def L1_norm(self, image):
        image_normalized = image / np.linalg.norm(image, ord=1, axis=(0, 1), keepdims=True)
        image_normalized *= 255 / np.max(image_normalized)
        return image_normalized

    def L2_norm(self, image):
        image_normalized = image / np.linalg.norm(image, ord=2, axis=(0, 1), keepdims=True)
        image_normalized *= 255 / np.max(image_normalized)
        return image_normalized
