import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    # Use skimage io.imread
    out = io.imread(image_path)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    image = image/255
    out = np.multiply(0.5, np.multiply(image, image))
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width)
    """
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = color.rgb2gray(image)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

