import numpy as np
import cv2
import matplotlib.image as mpimg

from car_detect import find_windows, process_car_candidates
from car_detect_svm import find_cars
from cnn import predict, predict_classes

class CarDetectorSvm(object):
    def __init__(self, classifier, X_scaler,
                 cs, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                 spatial_feat=True, hist_feat=True, hog_feat=True):
        self.svc = classifier
        self.X_scaler = X_scaler
        self.cs = cs
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins

        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    def find_car_candidates(self, img, wins):
        car_candidates_img, car_candidate_boxes = find_cars(img, wins, self.svc, self.X_scaler,
                                                            self.cs, self.orient, self.pix_per_cell,
                                                            self.cell_per_block, self.spatial_size, self.hist_bins,
                                                            spatial_feat=self.spatial_feat, hist_feat=self.hist_feat,
                                                            hog_feat=self.hog_feat)
        return car_candidates_img, car_candidate_boxes


class CarDetectorNN(object):
    def __init__(self, model, gray, hist_eq, img_size=(64, 64)):
        self.model = model
        self.gray = gray
        self.hist_eq = hist_eq
        self.img_size = img_size

    def find_car_candidates(self, img, wins):
        h, w = img.shape[0], img.shape[1]

        draw_img = np.copy(img)
        box_list = []

        test_imgs = []

        for win in wins:
            win_left_top, win_right_bottom = win
            (x_start, y_start) = win_left_top
            (x_stop, y_stop) = win_right_bottom

            # adjust x coordinates
            x_start = max(0, x_start)
            x_stop = min(w, x_stop)

            if (x_stop - x_start) <= 1 or (y_stop - y_start) <= 1:
                continue

            # Extract the image patch
            subimg = cv2.resize(img[y_start:y_stop, x_start:x_stop], self.img_size)
            test_imgs.append(subimg)

        classes = predict_classes(self.model, np.array(test_imgs), gray=self.gray, hist_eq=self.hist_eq)
        #classes = predict(self.model, np.array(test_imgs), gray=self.gray, hist_eq=self.hist_eq)
        for c, win in zip(classes, wins):
            if c == 0:
            #if c[0] >= 0.9:
                win_left_top, win_right_bottom = win

                # draw a box
                cv2.rectangle(draw_img, win_left_top, win_right_bottom, (0, 0, 255), 6)

                # add found box to list
                box_list.append((win_left_top, win_right_bottom))

        return draw_img, box_list


# Class which tracks vehicles in passed video frames
class VehicleTracker(object):
    def __init__(self, car_detector, average_model=True, heatmap_threshold=1):
        self.car_detector = car_detector
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
        car_candidates_img, car_candidate_boxes = self.car_detector.find_car_candidates(img, self.wins)

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

