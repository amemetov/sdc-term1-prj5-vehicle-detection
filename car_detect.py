import numpy as np
import cv2
import math

from features import bin_spatial, color_hist, get_hog_features


def convert_color(image, conv='YCrCb'):
    if conv == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif conv == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif conv == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif conv == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif conv == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(image)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def find_windows(img, y_start=280, x_overlap=0.5,
                 start_level_win_size=320, end_level_win_size=64, n_levels=6,
                 start_level_color=(255, 0, 0), end_level_color=(0, 255, 128),
                 visualize=True):
    draw_img = np.copy(img)

    h, w = img.shape[0], img.shape[1]

    # the list of detected windows in form ((x1, y1), (x2, y2))
    windows = []

    win_size_step = (start_level_win_size - end_level_win_size) // (n_levels - 1)
    y_step = win_size_step // 2
    color_step = (np.asarray(end_level_color) - np.asarray(start_level_color)) // (n_levels - 1)

    y = y_start
    win_size = start_level_win_size
    color = np.asarray(start_level_color)
    for l in range(n_levels):
        x_level_start = 0

        # Compute the span of the region to be searched
        xspan = w - x_level_start

        # Compute the number of pixels per step in x
        nx_pix_per_step = np.int(win_size * (1 - x_overlap))

        # Compute the number of windows in x
        nx_buffer = np.int(win_size * x_overlap)
        nx_wins = np.int((xspan - nx_buffer) / nx_pix_per_step)

        # Compute how many pixels are used
        x_total_win_size = nx_wins * nx_pix_per_step
        if x_overlap > 0:
            x_total_win_size += nx_pix_per_step

        # Shift windows to center if they do not fill the image along width
        if x_total_win_size < w:
            x_level_start = (w - x_total_win_size) // 2

        level_color = (int(color[0]), int(color[1]), int(color[2]))

        for xs in range(nx_wins):
            # Calculate window position
            x_start = xs * nx_pix_per_step + x_level_start
            x_end = x_start + win_size

            win_left_top = (x_start, y)
            win_right_bottom = (x_end, y + win_size)

            windows.append((win_left_top, win_right_bottom))

            if visualize:
                cv2.rectangle(draw_img, win_left_top, win_right_bottom, level_color, 6)


        # move to the next level
        y += y_step
        win_size -= win_size_step
        color += color_step

    print('Detected windows number: {0}'.format(len(windows)))
    return draw_img, windows

def find_cars(img, wins, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              patch_size=(64,64)):
    h, w = img.shape[0], img.shape[1]

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    (_, global_y_start), (_, global_y_stop) = wins[0]

    #img_cvt = convert_color(img[global_y_start:global_y_stop], conv=cspace)
    img_cvt = convert_color(img, conv=cspace)
    ch1 = img_cvt[:, :, 0]
    ch2 = img_cvt[:, :, 1]
    ch3 = img_cvt[:, :, 2]

    # Compute individual channel HOG features for the entire image
    #hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #print('ch1.shape: {0}'.format(ch1.shape))
    #print('hog1.shape: {0}'.format(hog1.shape))

    box_list = []

    for win in wins:
        win_left_top, win_right_bottom = win

        (x_start, y_start) = win_left_top
        (x_stop, y_stop) = win_right_bottom

        x_stop = min(x_stop, w)
        if x_start == x_stop:
            continue


        #hog_x_start = x_start // pix_per_cell

        # Extract the image patch
        subimg = cv2.resize(img_cvt[y_start:y_stop, x_start:x_stop], patch_size)

        # Extract HOG for this patch
        hog_feat1 = get_hog_features(subimg[:, :, 0], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
        hog_feat2 = get_hog_features(subimg[:, :, 1], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
        hog_feat3 = get_hog_features(subimg[:, :, 2], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)

        # Scale features and make a prediction
        test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        test_prediction = svc.predict(test_features)

        # search box
        if test_prediction == 1:
            # draw a box
            cv2.rectangle(draw_img, win_left_top, win_right_bottom, (0, 0, 255), 6)

            # add found box to list
            box_list.append((win_left_top, win_right_bottom))

    return draw_img, box_list


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars2(img, ystart, ystop, scale, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # "box" takes the form ((x1, y1), (x2, y2))
    box_list = []

    print('Blocks: {0}'.format(nxsteps*nysteps))

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            # search box
            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)

            box_left_top = (xbox_left, ytop_draw + ystart)
            box_right_bottom = (xbox_left + win_draw, ytop_draw + win_draw + ystart)

            if test_prediction == 1:
                # draw a box
                cv2.rectangle(draw_img, box_left_top, box_right_bottom, (0, 0, 255), 6)

                # add found box to list
                box_list.append((box_left_top, box_right_bottom))
            else:
                #print('Box {0}'.format((box_left_top, box_right_bottom)))
                #cv2.rectangle(draw_img, box_left_top, box_right_bottom, (255, 0, 0), 6)
                continue

    return draw_img, box_list
