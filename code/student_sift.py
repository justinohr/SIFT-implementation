import numpy as np
import cv2
import scipy.signal


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    fv = []
    offset = int(feature_width / 2)
    gaussian = cv2.getGaussianKernel(1, 7)
    image = scipy.signal.convolve(image, gaussian)
    graident_x = cv2.Sobel(image, 1,0)
    gradient_y = cv2.Sobel(image, 0,1)
    mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    phase = get_phase(gradient_x, gradient_y, gradient_x.size)

    for x_, y_ in zip(x, y):
        x_ = int(round(x_[0]))
        y_ = int(round(y_[0]))
        m = mag[y_-offset:y_+offset, x_-offset:x_+offset]
        p = phase[y_-offset:y_+offset, x_-offset:x_+offset]

        bins = np.zeros(36)

        for m_, p_ in zip(m,p):
            bins[int(p_/10)] += m_
        orientation = np.argmax(bins)

        p += orientation * 10
        p %= 360
        print(p)
        feature = getHistogram(m, p)
        feature = feature.reshape(feature.size)
        fv.append(feature)

    fv = np.array(fv)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv

# The range of phase is 0 to 2pi
def get_phase(grad_x, grad_y, N):
    phase = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if grad_x[i,j] > 0 and grad_y[i,j] > 0:
                phase[i,j] = np.arctan(np.division(gradient_y, gradient_x))
            elif grad_x[i,j] < 0 and grad_y[i,j] > 0:
                phase[i,j] = np.arctan(np.division(gradient_y, gradient_x)) + np.pi
            elif grad_x[i,j] < 0 and grad_y[i,j] < 0:
                phase[i,j] = np.arctan(np.division(gradient_y, gradient_x)) + np.pi
            else:
                phase[i,j] = np.arctan(np.division(gradient_y, gradient_x)) + np.pi * 2

    return phase