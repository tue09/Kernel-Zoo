import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

import kernel
from normalize import Normalize

kernel = kernel.kernels["negative"]
normalizer = Normalize("None")


def convolution(A, kernel, normalizer, padding=1, stride=1):
    print(f"kernel = {kernel}")
    kernel_size = kernel.shape[0]
    A = np.array([np.pad(sub_arr, pad_width=padding, mode='constant', constant_values=0) for sub_arr in A])
    img_shape = A.shape 
    
    output_height = int((img_shape[1] - kernel_size + 1) / stride)
    output_width = int((img_shape[2] - kernel_size + 1) / stride)
    print(f"height = {output_height}, width = {output_width}")

    C = np.zeros((output_height, output_width, 3))

    for k in range (0, 3):
        for i in range(0, output_height):
            for j in range(0, output_width):
                C[i, j, k] = np.sum(np.multiply(kernel, A[k, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]))

    C = normalizer._normalize_method(C)
    return C.astype('uint8')

if __name__ == "__main__":
    current_directory = os.getcwd()
    current_directory = current_directory.replace('\\','/')
    script_directory = os.path.dirname(os.path.realpath(__file__))
    script_directory = script_directory.replace('\\','/')

    print(f"Current directory = {current_directory}")
    print(f"Script directory = {script_directory}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    img = Image.open(script_directory + '/images/Happy_Duck.jpg')
    img_np_arr = np.asarray(img)
    print(f"shape root = {img_np_arr.shape}")
    ax1.imshow(img_np_arr, interpolation='nearest')
    ax1.set_title('Original Image')

    img_np_arr = img_np_arr.transpose((2, 0, 1))

    output = convolution(A=img_np_arr, kernel=kernel, normalizer=normalizer)
    print(f"shape = {output.shape}")
    ax2.imshow(output, interpolation='nearest')
    ax2.set_title('Convolution Image')

    plt.show()
    pass