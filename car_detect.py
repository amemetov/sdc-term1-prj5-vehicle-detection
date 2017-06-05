import numpy as np
import cv2
from scipy.ndimage.measurements import label


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


# Filter out False Positives and Multiple Detections by using Heatmap
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


# Build a list of windows where a classifier will search cars
def find_windows(img, y_start=360, y_stop=400,
                 dx_start=0, dx_stop=100,
                 x_overlap_start=0.1, x_overlap_stop=0.9,
                 start_level_win_size=260, end_level_win_size=80,
                 n_levels=4,
                 start_level_color=(255, 0, 0), end_level_color=(0, 255, 128),
                 visualize=True):
    draw_img = np.copy(img)

    h, w = img.shape[0], img.shape[1]

    # the list of detected windows in form ((x1, y1), (x2, y2))
    windows = []

    win_size_step = (start_level_win_size - end_level_win_size) // (n_levels - 1)
    y_step = (y_stop - y_start) // (n_levels - 1)
    dx_step = (dx_stop - dx_start) // (n_levels - 1)
    x_overlap_step = (x_overlap_stop - x_overlap_start) / (n_levels - 1)
    color_step = (np.asarray(end_level_color) - np.asarray(start_level_color)) // (n_levels - 1)

    y = y_start
    dx = dx_start
    win_size = start_level_win_size
    x_overlap = x_overlap_start
    color = np.asarray(start_level_color)


    for l in range(n_levels):
        x_level_start = dx

        # Compute the span of the region to be searched
        xspan = w - 2*x_level_start

        # Compute the number of pixels per step in x
        nx_pix_per_step = np.int(win_size * (1 - x_overlap))

        # Compute the number of windows in x
        nx_buffer = np.int(win_size * x_overlap)
        nx_wins = np.int((xspan - nx_buffer) / nx_pix_per_step) + 1

        total_pixels_per_level = nx_wins*nx_pix_per_step
        if total_pixels_per_level > xspan:
            x_level_start -= (total_pixels_per_level - xspan) // 2

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
        dx += dx_step
        win_size -= win_size_step
        x_overlap += x_overlap_step
        color += color_step

    print('Detected windows number: {0}'.format(len(windows)))
    return draw_img, windows