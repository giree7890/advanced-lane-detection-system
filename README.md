# Advanced Lane Detection with Steering and Departure Analysis

This project is an implementation of a computer vision-based lane detection system. It processes a video stream from a file or a live camera to identify lane lines on the road. The system then calculates the lane's curvature and the vehicle's position relative to the center of the lane, providing real-time feedback by overlaying the detected lane and relevant metrics onto the video feed.

This is a common task in the development of Advanced Driver-Assistance Systems (ADAS) and autonomous vehicles.

## Features

- **Lane Line Detection**: Identifies and tracks the left and right lane lines.
- **Perspective Transformation**: Creates a bird's-eye view of the road for accurate lane analysis.
- **Lane Curvature Calculation**: Measures the radius of curvature of the road in meters.
- **Vehicle Position Analysis**: Determines the vehicle's offset from the center of the lane (lane departure).
- **Real-time Visual Feedback**: Overlays the detected lane, curvature, and offset information onto the output video stream.
- **Flexible Input**: Works with both pre-recorded video files and live camera feeds.

## How It Works

The lane detection process is implemented as a pipeline of image processing steps:

1.  **Image Pre-processing (`process_image`)**:
    - The input video frame is converted from BGR to the HLS (Hue, Lightness, Saturation) color space.
    - A color mask is applied to isolate white and yellow-ish pixels, which are characteristic of lane lines.
    - The image is converted to grayscale.
    - Gaussian blur is applied to reduce noise and smooth the image.
    - The Canny edge detection algorithm is used to identify potential edges in the image.

2.  **Perspective Transformation (`perspective_warp`)**:
    - A region of interest (a trapezoid in the camera's view) is selected.
    - This region is transformed into a rectangular "bird's-eye view". This unwraps the perspective, making the lane lines appear parallel and simplifying further analysis.

3.  **Lane Pixel Identification (`sliding_window_search`)**:
    - A histogram of the bottom half of the warped image is created to find the starting positions of the left and right lanes.
    - A "sliding window" approach is used to move up the image, following the lane lines and identifying all the pixels belonging to each lane.
    - A second-order polynomial (`y = Ax² + Bx + C`) is fitted to the identified pixels for both the left and right lanes. This mathematically represents the shape of the lanes.

4.  **Measurement and Calculation**:
    - **Curvature (`measure_curvature`)**: The fitted polynomials are used to calculate the radius of curvature. The pixel values are first converted to real-world measurements (meters) before the calculation.
    - **Vehicle Offset (`off_center`)**: The center of the lane is determined by averaging the positions of the left and right lane fits at the bottom of the screen. This is compared to the center of the image to calculate the vehicle's deviation from the lane center in meters.

5.  **Visualization (`draw_lane_lines` & `add_text`)**:
    - A polygon is drawn to represent the area between the detected lane lines.
    - This polygon is warped back from the bird's-eye view to the original camera perspective.
    - The resulting lane area is overlaid onto the original video frame.
    - The calculated radius of curvature and vehicle offset are displayed as text on the final video output.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone <your-repo-url>
    cd lane-detection-with-steer-and-departure-master
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install opencv-python numpy
    ```

## Usage

The script can be configured to use either a video file or a live camera feed.

### Using a Video File

1.  Place your video file (e.g., `sample1.mp4`) inside a `videos` directory in the project root.
2.  In `cameraCode.py`, make sure the video file input is active:

    ```python
    # Use a video file as input
    cap = cv2.VideoCapture("./videos/sample1.mp4")
    ```

3.  Run the script:
    ```bash
    python cameraCode.py
    ```

### Using a Live Camera

1.  In `cameraCode.py`, comment out the video file line and uncomment the camera line:

    ```python
    # Use the camera (uncomment the line below for live camera feed)
    cap = cv2.VideoCapture(0)

    # Use a video file as input (comment this line if using the camera)
    # cap = cv2.VideoCapture("./videos/sample1.mp4")
    ```
    *Note: `0` is typically the default built-in webcam. If you have multiple cameras, you might need to use `1`, `2`, etc.*

2.  Run the script:
    ```bash
    python cameraCode.py
    ```

While the video is playing, press the **'q'** key to quit the application.

## Configuration and Tuning

The performance of the lane detection can be sensitive to lighting conditions, road textures, and camera angles. You may need to adjust the following parameters in `cameraCode.py` for optimal results with different videos or cameras:

- **Color Thresholds (`process_image`)**: The `lower_white` and `upper_white` NumPy arrays can be adjusted to better isolate lane lines in different lighting.
- **Perspective Warp Points (`perspective_warp`)**: The `src` and `dst` points are crucial. The `src` points must be carefully selected to form a trapezoid that captures the lane area just in front of the car. These are hardcoded and will likely need to be recalibrated for a different camera setup or video resolution.
- **Sliding Window Parameters (`sliding_window_search`)**: Parameters like `nwindows`, `margin`, and `minpix` can be tuned to improve the robustness of the lane finding algorithm.
