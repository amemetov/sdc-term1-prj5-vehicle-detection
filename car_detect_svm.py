import numpy as np
import cv2
import math
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from features import convert_color, bin_spatial, color_hist, get_hog_features, extract_features

# Train linear SVC on passed images using passed params to build features
def train(vehicles_files, not_vehicles_files,
          cs, spatial_size, hist_bins,
          orient, pix_per_cell, cell_per_block, hog_channel,
          spatial_feat=True, hist_feat=True, hog_feat=True):

    start_time = time.time()

    test_cars = vehicles_files
    test_not_cars = not_vehicles_files

    car_features = extract_features(test_cars, cs, spatial_size, hist_bins,
                                    orient, pix_per_cell, cell_per_block, hog_channel,
                                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                    vis_hog=False)

    not_car_features = extract_features(test_not_cars, cs, spatial_size, hist_bins,
                                        orient, pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                        vis_hog=False)

    spent_time = time.time() - start_time
    print('Feature extraction spent time {0:.2f} seconds'.format(spent_time))

    X = np.vstack((car_features, not_car_features)).astype(np.float64)

    # Normalize data
    # fit a per column scaler
    X_scaler = StandardScaler().fit(X)
    # apply the scaler to X
    X_scaled = X_scaler.transform(X)

    # define the labels vector y
    # 1 - is the label for Car
    # 0 - is the label for Not Car
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # split the data into train and test sets
    # train_test_split splits arrays into random train and test subsets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))

    # Train using LinearSVC (Linear Support Vector Classification)
    start_time = time.time()
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    spent_time = time.time() - start_time
    print('Training spent time {0:.2f} seconds'.format(spent_time))

    # check the test accuracy of the model
    test_accuracy = svc.score(X_test, y_test)
    print('Test Accuracy: {0:.3f}'.format(test_accuracy))

    return svc, X_scaler



def find_cars(img, wins, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              spatial_feat=True, hist_feat=True, hog_feat=True, patch_size=(64, 64)):
    h, w = img.shape[0], img.shape[1]

    draw_img = np.copy(img)

    # image should be in the range [0, 1]
    img = img.astype(np.float32) / 255

    # convert to expected color space
    img_cvt = convert_color(img, cspace=cspace)

    # the result box list
    box_list = []

    last_y_start, last_y_stop = None, None
    hog1, hog2, hog3 = None, None, None
    for win in wins:
        win_features = []

        win_left_top, win_right_bottom = win
        (x_start, y_start) = win_left_top
        (x_stop, y_stop) = win_right_bottom

        # adjust x coordinates
        x_start = max(0, x_start)
        x_stop = min(w, x_stop)

        if (x_stop - x_start) <= 1 or (y_stop - y_start) <= 1:
            continue

        # Extract the image patch
        subimg = cv2.resize(img_cvt[y_start:y_stop, x_start:x_stop], patch_size)

        # Get color features
        if spatial_feat:
            spatial_features = bin_spatial(subimg, size=spatial_size)
            win_features.append(spatial_features)

        if hist_feat:
            hist_features = color_hist(subimg, nbins=hist_bins)
            win_features.append(hist_features)

        if hog_feat and last_y_start != y_start or last_y_stop != y_stop:
            last_y_start, last_y_stop = y_start, y_stop
            # new level detected - rebuild HOG features
            # Compute individual channel HOG features for the entire image
            img_stride = img_cvt[y_start:y_stop, :]
            img_stride_h = patch_size[0]
            scale = (y_stop - y_start) / img_stride_h
            img_stride_w = np.int(w / scale)
            img_stride = cv2.resize(img_stride, (img_stride_w, img_stride_h))

            ch1 = img_stride[:, :, 0]
            ch2 = img_stride[:, :, 1]
            ch3 = img_stride[:, :, 2]

            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        if hog_feat:
            # Extract HOG for this patch
            x_hog_start = np.int(math.floor((hog1.shape[1] / w) * x_start))
            x_hog_step = hog1.shape[0]
            x_hog_stop = x_hog_start + x_hog_step
            if x_hog_stop > hog1.shape[1]:
                x_hog_stop = hog1.shape[1]
                x_hog_start = x_hog_stop - x_hog_step

            hog_feat1 = hog1[:, x_hog_start:x_hog_stop].ravel()
            hog_feat2 = hog2[:, x_hog_start:x_hog_stop].ravel()
            hog_feat3 = hog3[:, x_hog_start:x_hog_stop].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            win_features.append(hog_features)

        # Scale features and make a prediction
        #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        test_features = X_scaler.transform(np.concatenate(win_features).reshape(1, -1))
        test_prediction = svc.predict(test_features)

        if test_prediction == 1:
            # draw a box
            cv2.rectangle(draw_img, win_left_top, win_right_bottom, (0, 0, 255), 6)

            # add found box to list
            box_list.append((win_left_top, win_right_bottom))

    return draw_img, box_list




# # Define a single function that can extract features using hog sub-sampling and make predictions
# def find_cars2(img, ystart, ystop, scale, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
#     draw_img = np.copy(img)
#     img = img.astype(np.float32) / 255
#
#     img_tosearch = img[ystart:ystop, :, :]
#     ctrans_tosearch = convert_color(img_tosearch, conv=cspace)
#     if scale != 1:
#         imshape = ctrans_tosearch.shape
#         ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
#
#     ch1 = ctrans_tosearch[:, :, 0]
#     ch2 = ctrans_tosearch[:, :, 1]
#     ch3 = ctrans_tosearch[:, :, 2]
#
#     # Define blocks and steps as above
#     nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
#     nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
#     nfeat_per_block = orient * cell_per_block ** 2
#
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
#     cells_per_step = 2  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step
#
#     # Compute individual channel HOG features for the entire image
#     hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#     hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
#
#     # "box" takes the form ((x1, y1), (x2, y2))
#     box_list = []
#
#     print('Blocks: {0}'.format(nxsteps*nysteps))
#
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb * cells_per_step
#             xpos = xb * cells_per_step
#             # Extract HOG for this patch
#             hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
#             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
#
#             xleft = xpos * pix_per_cell
#             ytop = ypos * pix_per_cell
#
#             # Extract the image patch
#             subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
#
#             # Get color features
#             spatial_features = bin_spatial(subimg, size=spatial_size)
#             hist_features = color_hist(subimg, nbins=hist_bins)
#
#             # Scale features and make a prediction
#             test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
#             test_prediction = svc.predict(test_features)
#
#             # search box
#             xbox_left = np.int(xleft * scale)
#             ytop_draw = np.int(ytop * scale)
#             win_draw = np.int(window * scale)
#
#             box_left_top = (xbox_left, ytop_draw + ystart)
#             box_right_bottom = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
#
#             if test_prediction == 1:
#                 # draw a box
#                 cv2.rectangle(draw_img, box_left_top, box_right_bottom, (0, 0, 255), 6)
#
#                 # add found box to list
#                 box_list.append((box_left_top, box_right_bottom))
#             else:
#                 #print('Box {0}'.format((box_left_top, box_right_bottom)))
#                 #cv2.rectangle(draw_img, box_left_top, box_right_bottom, (255, 0, 0), 6)
#                 continue
#
#     return draw_img, box_list
#
#
