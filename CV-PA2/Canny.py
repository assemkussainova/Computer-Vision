import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding as used in the previous assignment can make
    # derivatives at the image boundary very big.
    
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge') 

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    for c in range(Wi):
        for r in range(Hi):
            #create indexes on new image with padding
            x = c + pad_width1
            y = r + pad_width0
            new_image = padded[y - pad_width0:y + pad_width0 + 1,
                               x - pad_width1:x + pad_width1 + 1]
            conv_out = np.sum(np.multiply(new_image, kernel))
            out[r, c] = conv_out
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    k = int(size/2)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*(sigma**2))) * np.exp(-1*((i-k)**2 + (j-k)**2)/(2*(sigma**2)))
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function which is defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    k = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    out = conv(img, k)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function which is defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    k = np.array([[0, 0.5, 0], [0, 0, 0], [0, -0.5, 0]])
    out = conv(img, k)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    Gx, Gy = partial_x(img), partial_y(img)
    G = ((Gx)**2 + (Gy)**2)**(1/2)

    theta = np.arctan2(Gy, Gx)*180/np.pi
    theta = (theta+360)%360
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return G, theta

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    theta[theta == 360] = 0
    neighbors = dict()

    for angle in range(0, 360, 45):
        if angle == 0 or angle == 180:
            val = [[0, 1], [0, -1]]
        elif angle == 45 or angle == 225:
            val = [[1, 1], [-1, -1]]
        elif angle == 90 or angle == 270:
            val = [[1, 0], [-1, 0]]
        elif angle == 135 or angle == 315:
            val = [[1, -1], [-1, 1]]
        neighbors[angle] = val

    for i in range(1, H-1):
        for j in range(1, W-1):
            pos, neg = neighbors[theta[i, j]]

            i_pos = i + pos[0]
            j_pos = j + pos[1]

            i_neg = i + neg[0]
            j_neg = j + neg[1]

            if G[i, j] >= G[i_pos, j_pos] and G[i, j] >= G[i_neg, j_neg]:
                out[i, j] = G[i, j]
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    strong_edges = img >= high
    weak_edges = (img < high) & (img > low)

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    check = []

    for i in range(len(indices)):
        check.append((indices[i, 0], indices[i, 1]))

    while len(check) != 0:
        (y, x) = check.pop(0)
        neighbors = get_neighbors(y, x, H, W)

        for j in range(len(neighbors)):
            if weak_edges[neighbors[j][0], neighbors[j][1]]:
                edges[neighbors[j][0], neighbors[j][1]] = True
                check.append((neighbors[j][0], neighbors[j][1]))
                weak_edges[neighbors[j][0], neighbors[j][1]] = False
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return edges

