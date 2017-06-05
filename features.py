import matplotlib.image as mpimg
import numpy as np
import cv2

from skimage.feature import hog

# Compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Return HOG features for specified channel
def get_hog_features_for_channel(feature_image, hog_channel, orient, pix_per_cell, cell_per_block, vis=False):
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=vis, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block, vis=vis, feature_vec=True)

    return hog_features

# Extract features from a list of images
def extract_features(imgs, cspace='RGB',
                     spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True, vis_hog=False):
    # Create a list to append feature vectors to
    features = []
    hog_images = []

    # Iterate through the list of images
    for file in imgs:
        feature_image = create_feature_image(file, cspace)

        file_features = []

        if spatial_feat == True:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Apply get_hog_features_for_channel() to get HOG features
            hog_features = get_hog_features_for_channel(feature_image, hog_channel, orient, pix_per_cell, cell_per_block, vis=vis_hog)
            if vis_hog == True:
                hog_features, hog_img = hog_features
                hog_images.append(hog_img)

            file_features.append(hog_features)

        # Append the new feature vector to the features list
        features.append(np.concatenate(file_features))


    if vis_hog == True:
        return features, hog_images

    return features


# Convert image to specified color space and return it as is
def create_feature_image(img_file, cspace='RGB'):
    # Read in each one by one
    image = mpimg.imread(img_file)

    # apply color conversion
    feature_image = convert_color(image, cspace)

    return feature_image


# Convert to specified color space
def convert_color(image, cspace='YCrCb'):
    if cspace == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(image)