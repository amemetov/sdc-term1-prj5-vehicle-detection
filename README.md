# **Vehicle Detection** 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car-not-car]: ./output_images/car-not-car.png
[hog-features-orient-6]: ./output_images/hog-features-orient-6.png
[hog-features-orient-9]: ./output_images/hog-features-orient-9.png
[hog-features-pix-per-cell-4]: ./output_images/hog-features-pix-per-cell-4.png
[hog-features-pix-per-cell-16]: ./output_images/hog-features-pix-per-cell-16.png
[hog-features-cell-per-block-4]: ./output_images/hog-features-cell-per-block-4.png
[wins-sliding]: ./output_images/wins-sliding.png
[classifier-pipeline1]: ./output_images/classifier-pipeline1.png
[classifier-pipeline2]: ./output_images/classifier-pipeline2.png
[classifier-pipeline3]: ./output_images/classifier-pipeline3.png
[whole-pipeline1]: ./output_images/whole-pipeline1.png
[whole-pipeline2]: ./output_images/whole-pipeline2.png
[whole-pipeline3]: ./output_images/whole-pipeline3.png
[video1]: ./output_videos/project_video_out_notavg.mp4
[video2]: ./output_videos/project_video_out_avg.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Submitted Files

#### 1. Submission includes following files:

* [P5-Vehicle-Detection.ipynb](P5-Vehicle-Detection.ipynb) IPython notebook file containing workflow of the project
* [car_tracker.py](car_tracker.py) containing class which encapsulates tracking cars for video frames
* [features.py](features.py) containing methods for creating features for classifier
* [car_detect_svm.py](car_detect_svm.py) containing methods for training a classifier and finding car candidates
* [car_detect.py](car_detect.py) containing methods for sliding windows, heatmaps, solvers for False Positives and Multiple Detections
* [cnn.py](cnn.py) containing methods to build CNN for classifying cars/not-cars
* [project_video_out_notavg.mp4](project_video_out_notavg.mp4) the result output video
* [project_video_out_avg.mp4](project_video_out_avg.mp4) the result output video where current bounding boxes are averaged with recent bounding boxes


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file [features.py](features.py) method `extract_features` (line #55).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car-not-car]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like for different color spaces 
and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog-features-orient-6]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`):


`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
![alt text][hog-features-orient-9]

I noticed that `orientations=9` gives more precise gradients (than for `orientations=6`)  which allow to notice a car (especially for `YCrCb` color space) 


Then I tried to tune `pixels_per_cell`:

`orientations=9`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`
![alt text][hog-features-pix-per-cell-4]

`orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`
![alt text][hog-features-pix-per-cell-16]

As for me `pixels_per_cell=8` produces optimal result.


For `cells_per_block` I did not notice a big difference

`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)`
![alt text][hog-features-cell-per-block-4]


So I decided to stop on values:

`color_space=YCrCb`, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training classifier is contained in the file [car_detect_svm.py](car_detect_svm.py) method `train` (line #13).
Firstly I tried different values for `hog_channel` and noticed that using `ALL` channels gives a better result than using any single channel separatelly.
Then I tried to include/exclude using `spatial_feature` and `hist_feateture`.
Including all of them gives better result.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have implemented the idea shown in the lecture `#33. Multi-scale Windows`.

The code is contained in the file [car_detect.py](car_detect.py) method `find_windows` (line #74).

The result with no overlapping is shown in the below image:

![alt text][wins-sliding]

The method have several parameters which specify:

* `n_levels` - How many levels of windows should be constructed.
* `(y_start, y_stop)` - Which Y coordinates should be chosen for the first level of windows and for the final level.
* `(dx_start, dx_stop)` - How many pixels along X axis windows should be shifted from boundaries respectivelly for the first level of windows and for the final level.
* `(x_overlap_start, x_overlap_stop)` - How much windows should be overlapped respectivelly for the first level of windows and for the final level.
* `(start_level_win_size, end_level_win_size)` - The size of windows respectivelly for the first level of windows and for the final level.

For intermediate levels the necessary parameters are interpolated depending on the level position.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
Here are some example images:

![alt text][classifier-pipeline1]

![alt text][classifier-pipeline2]

![alt text][classifier-pipeline3]

Mostly for optimizing the result of classifying I tuned sliding window method to get good result on test images.
For `SVM` itself I used `GridSearchCV` to find the best parameters ([P5-Vehicle-Detection.ipynb](P5-Vehicle-Detection.ipynb) cell #8).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video_out_avg.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To solve `False Positives` and `Multiple Detections` I have used Heatmap.

The code is contained in the file [car_detect.py](car_detect.py) method `process_car_candidates` (line #53).

The method takes the positions of positive detections.

Then method creates a heatmap from the positive detections and then thresholds that map to identify vehicle positions.  

Then method uses `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  

It is assumed that each blob corresponds to a vehicle.

The final step - constructing bounding boxes to cover the area of each blob detected.  

Here's an example result showing the whole pipeline:

![alt text][whole-pipeline1]

![alt text][whole-pipeline2]

![alt text][whole-pipeline3]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It's my first project where I do feature engineering for computer vision task.

Each step required a lot of tunings to get appropriate result.

And even after all done work I'm not very satisfied with the final result.

I tried to use `CNN` ([cnn.py](cnn.py)) to see how it can help, but to tell the truth, built CNN produced a lot of False Postives and False Negatives (for Cars).
I think that collecting more train data (especially from test video) can improve CNN's performance.

I found that there are many Deep Learning approaches which may give much better result than HOG:
* [R-CNN](https://arxiv.org/abs/1311.2524)
* [Fast R-CNN](https://arxiv.org/abs/1504.08083)
* [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [Mask R-CNN](https://arxiv.org/abs/1703.06870)
* [OverFeat](https://arxiv.org/abs/1312.6229)
* [U-Net](https://arxiv.org/abs/1505.04597)
* [YOLO9000](https://arxiv.org/abs/1612.08242)



