import numpy as np
import cv2
import math
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg

from features import convert_color, bin_spatial, color_hist, get_hog_features


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def get_labeled_bboxes(img, labels):
    draw_img = np.copy(img)

    # the result boxes
    boxes = []

    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # append defined bounding box to the result list
        boxes.append(bbox)

        # Draw the box on the image
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return draw_img, boxes


def process_car_candidates(img, car_candidate_boxes, heatmap_threshold=1):
    # Create heat map
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, car_candidate_boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heatmap_threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, car_boxes = get_labeled_bboxes(img, labels)

    return heatmap, draw_img, car_boxes


def find_windows(img, y_start=360, y_stop=400,
                 x_overlap_start=0.2, x_overlap_stop=0.8,
                 start_level_win_size=260, end_level_win_size=80, n_levels=8,
                 start_level_color=(255, 0, 0), end_level_color=(0, 255, 128),
                 visualize=True):
    draw_img = np.copy(img)

    h, w = img.shape[0], img.shape[1]

    # the list of detected windows in form ((x1, y1), (x2, y2))
    windows = []

    win_size_step = (start_level_win_size - end_level_win_size) // (n_levels - 1)
    y_step = (y_stop - y_start) // (n_levels - 1)
    x_overlap_step = (x_overlap_stop - x_overlap_start) / (n_levels - 1)
    color_step = (np.asarray(end_level_color) - np.asarray(start_level_color)) // (n_levels - 1)

    y = y_start
    win_size = start_level_win_size
    x_overlap = x_overlap_start
    color = np.asarray(start_level_color)


    for l in range(n_levels):
        x_level_start = 0

        # Compute the span of the region to be searched
        xspan = w - x_level_start

        # Compute the number of pixels per step in x
        nx_pix_per_step = np.int(win_size * (1 - x_overlap))

        # Compute the number of windows in x
        nx_buffer = np.int(win_size * x_overlap)
        nx_wins = np.int((xspan - nx_buffer) / nx_pix_per_step) + 1

        # Compute how many pixels are used
        # x_total_win_size = nx_wins * nx_pix_per_step
        # if x_overlap > 0:
        #     x_total_win_size += nx_pix_per_step
        #
        # # Shift windows to center if they do not fill the image along width
        # if x_total_win_size < w:
        #     x_level_start = (w - x_total_win_size) // 2

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
        x_overlap += x_overlap_step
        color += color_step

    print('Detected windows number: {0}'.format(len(windows)))
    return draw_img, windows


def find_cars(img, wins, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              patch_size=(64, 64)):
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
        win_left_top, win_right_bottom = win
        (x_start, y_start) = win_left_top
        (x_stop, y_stop) = win_right_bottom

        if last_y_start != y_start or last_y_stop != y_stop:
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

        # Extract HOG for this patch
        x_hog_start = np.int(math.floor((hog1.shape[1] / w)*x_start))
        x_hog_step = hog1.shape[0]
        x_hog_stop = x_hog_start + x_hog_step
        if x_hog_stop > hog1.shape[1]:
            x_hog_stop = hog1.shape[1]
            x_hog_start = x_hog_stop - x_hog_step

        hog_feat1 = hog1[:, x_hog_start:x_hog_stop].ravel()
        hog_feat2 = hog2[:, x_hog_start:x_hog_stop].ravel()
        hog_feat3 = hog3[:, x_hog_start:x_hog_stop].ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

        # Extract the image patch
        subimg = cv2.resize(img_cvt[y_start:y_stop, x_start:x_stop], patch_size)

        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)

        # Scale features and make a prediction
        test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        test_prediction = svc.predict(test_features)

        if test_prediction == 1:
            # draw a box
            cv2.rectangle(draw_img, win_left_top, win_right_bottom, (0, 0, 255), 6)

            # add found box to list
            box_list.append((win_left_top, win_right_bottom))

    return draw_img, box_list


# Class which tracks vehicles in passed video frames
class VehicleTracker(object):
    def __init__(self, classifier, X_scaler,
                 cs, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                 average_model=True, heatmap_threshold=1):
        self.svc = classifier
        self.X_scaler = X_scaler
        self.cs = cs
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.heatmap_threshold = heatmap_threshold
        self.average_model = average_model

        self.wins = None
        self.car_boxes = []
        self.cnt = 0


    def process_image(self, img):
        #img = np.copy(img)
        #mpimg.imsave('./test_images/1/{0}.jpg'.format(self.cnt), img, format='jpg')
        self.cnt += 1

        if self.wins is None:
            wins_img, self.wins = find_windows(img, visualize=False)

        # find Boxes which may contain cars
        car_candidates_img, car_candidate_boxes = find_cars(img, self.wins, self.svc, self.X_scaler,
                                      self.cs, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)

        # process Multiple Detections and False Positives
        heatmap, final_img, car_boxes = process_car_candidates(img, car_candidate_boxes, self.heatmap_threshold)

        # build average model depending on specified param
        if self.average_model:
            # Make average boxes
            average_car_boxes = self.average_cars(car_boxes, self.car_boxes)

            result_img = self.draw_car_boxes(img, average_car_boxes)
        else:
            result_img = self.draw_car_boxes(img, car_boxes)

        # update recent car boxes
        self.car_boxes = car_boxes

        # if self.cnt >= 612 and self.cnt <= 614:
        #     print('----------------------------------------------------')
        #     # img = np.copy(img)
        #     mpimg.imsave('./test_images/2/{0}.jpg'.format(self.cnt), img, format='jpg')
        #     mpimg.imsave('./test_images/2/{0}-cand.jpg'.format(self.cnt), car_candidates_img, format='jpg')
        #     mpimg.imsave('./test_images/2/{0}-heatmap.jpg'.format(self.cnt), heatmap, format='jpg')
        #     mpimg.imsave('./test_images/2/{0}-final_img.jpg'.format(self.cnt), final_img, format='jpg')
        #
        #     print('cnt: {0}'.format(self.cnt))
        #     print('car_candidate_boxes: {0}'.format(car_candidate_boxes))
        #     print('car_boxes: {0}'.format(car_boxes))
        #     print('average_car_boxes: {0}'.format(average_car_boxes))

        return result_img



    def average_cars(self, new_car_boxes, recent_car_boxes):
        average_cars = []
        for new_car in new_car_boxes:
            max_overlapped_box = None
            max_overlapped_box_area = 0
            for prev_car in recent_car_boxes:
                overlap_area = self.get_overlap_area(new_car, prev_car)
                if overlap_area > max_overlapped_box_area:
                    max_overlapped_box = prev_car
                    max_overlapped_box_area = overlap_area

            new_car_area = self.get_area(new_car)
            # check whether it's the same car
            if max_overlapped_box is not None and max_overlapped_box_area >= 0.5*new_car_area:
                average_cars.append(self.average_boxes(new_car, max_overlapped_box))
            else:
                # not found recent car, just add new car as is
                average_cars.append(new_car)

        return average_cars

    def get_overlap_area(self, box1, box2):
        (x11, y11), (x12, y12) = box1
        (x21, y21), (x22, y22) = box2

        x_overlap = max(0, min(x12, x22) - max(x11, x21))
        y_overlap = max(0, min(y12, y22) - max(y11, y21))

        return x_overlap * y_overlap

    def get_area(self, box):
        (x1, y1), (x2, y2) = box
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return w*h

    def average_boxes(self, box1, box2):
        (x11, y11), (x12, y12) = box1
        (x21, y21), (x22, y22) = box2

        x1, y1 = (x11 + x21) // 2, (y11 + y21) // 2
        x2, y2 = (x12 + x22) // 2, (y12 + y22) // 2

        return ((x1, y1), (x2, y2))


    def draw_car_boxes(self, img, car_boxes):
        draw_img = np.copy(img)

        for bbox in car_boxes:
            # Draw the box on the image
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 0, 255), 6)

        return draw_img



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
