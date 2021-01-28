import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        #####################################
        #       START YOUR CODE HERE        #
        #####################################
        for i in range(N):
            distances = np.linalg.norm(features[i] - centers, axis=1)
            assignments[i] = np.argmin(distances) #Assign each point to the closest center
        newCenters = np.zeros((k, D))
        for idx in range(k):
            index = np.where(assignments == idx)
            newCenters[idx] = np.mean(features[index], axis=0) #Compute new center of each cluster
        if np.allclose(newCenters, centers):
            break #Stop if cluster assignments did not change
        else:
            centers = newCenters
        ######################################
        #        END OF YOUR CODE            #
        ######################################

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of vectorization and broadcasting with numpy functions to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        #####################################
        #       START YOUR CODE HERE        #
        #####################################
        featuresNew = np.tile(features, (k, 1))
        centersNew = np.repeat(centers, N, axis=0)

        distances = np.linalg.norm(featuresNew - centersNew, axis=1).reshape(k, N)
        assignments = np.argmin(distances, axis=0) #Assign each point to the closest center
        newCenters = np.zeros((k, D))
        for idx in range(k):
            index = np.where(assignments == idx)
            newCenters[idx] = np.mean(features[index], axis=0) #Compute new center of each cluster
        if np.allclose(newCenters, centers):
            break #Stop if cluster assignments did not change
        else:
            centers = newCenters
        ######################################
        #        END OF YOUR CODE            #
        ######################################

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    features = np.reshape(img, (H*W, C))
    ######################################
    #        END OF YOUR CODE            #
    ######################################


    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).
    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    position = np.dstack(np.mgrid[0:H, 0:W]).reshape((H*W, 2))
    features[:, 0:C] = np.reshape(color, (H*W, C))
    features[:, C:C+2] = position #Add position to pixels
    features = (features - np.mean(features, axis=0))/(np.std(features, axis=0)) #Normalize features
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return features

   

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    accuracy = np.mean(mask == mask_gt)
    ######################################
    #        END OF YOUR CODE            #s
    ######################################
    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments. 
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
