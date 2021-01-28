import numpy as np


def conv_naive(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for r in range(Hi):
        for c in range(Wi):
            conv_out = 0
            for i in range(Hk):
                for j in range(Wk):
                    if r+1-i < 0 or c+1-j < 0 or r+1-i >= Hi or c+1-j >= Wi: #check the borders of image
                        conv_out = conv_out + 0
                    else:
                        conv_out = conv_out + image[r+1-i][c+1-j] * kernel[i][j]
            out[r][c] = conv_out
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Example: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = np.zeros((H + 2*pad_height, W + 2*pad_width))
    out[pad_height:H + pad_height, pad_width:W + pad_width] = image #add image to the output
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    pad_height = int(Hk/2)
    pad_width = int(Wk/2)

    img_pad = zero_pad(image, pad_height, pad_width)

    for c in range(Wi):
        for r in range(Hi):
            #create indexes on new image with padding
            x = c + pad_width
            y = r + pad_height
            new_image = img_pad[y - pad_height:y + pad_height + 1,
                                x - pad_width:x + pad_width + 1]
            conv_out = np.sum(np.multiply(new_image, kernel))
            out[r, c] = conv_out
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

