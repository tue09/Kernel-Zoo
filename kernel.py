import numpy as np

kernels = {
    "identity": np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),

    "sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]]),

    "negative": np.array([[0, 0, 0],
                          [0, -1, 0],
                          [0, 0, 0]]),

    "outline": np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]]),    

    "box_blur": np.array([[1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9]]),                

    "gaussian_blur": np.array([[1/16, 1/8, 1/16],
                          [1/8, 1/4, 1/8],
                          [1/16, 1/8, 1/16]]),
                        
    "emboss": np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]]),

    "sobel_x": np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]),

    "sobel_y": np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]),

    "laplacian": np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])                        
}
