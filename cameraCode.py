import cv2
import numpy as np
import warnings

# Use the camera (uncomment the line below for live camera feed)
# cap = cv2.VideoCapture(0)

# Use a video file as input (comment this line if using the camera)
cap = cv2.VideoCapture("./videos/sample1.mp4")

# Conversion factors for pixels to meters
ym_per_pix = 30 / 720  # Meters per pixel in the vertical direction
xm_per_pix = 3.7 / 700  # Meters per pixel in the horizontal direction


def region_of_interest(img, vertices):
    # Create a black mask with the same size as the input image
    mask = np.zeros_like(img)
    # Fill the mask with white in the region defined by the vertices
    cv2.fillPoly(mask, vertices, 255)
    # Keep only the region of interest in the image
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines):
    # If lines are detected, draw them on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)


def process_image(frame):
    # Convert the image to HLS color space
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # Define the range for white color
    lower_white = np.array([0, 160, 0])
    upper_white = np.array([255, 255, 255])
    # Create a mask to filter out white areas
    mask = cv2.inRange(hls, lower_white, upper_white)
    # Apply the mask to the original image
    hls_result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to highlight lane lines
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    # Detect edges using the Canny edge detection algorithm
    edges = cv2.Canny(blur, 50, 150)

    return edges


def perspective_warp(img):
    # Get the size of the image
    img_size = (img.shape[1], img.shape[0])
    # Define points for the perspective transformation
    src = np.float32([[590, 440], [690, 440], [200, 640], [1000, 640]])
    dst = np.float32([[200, 0], [1200, 0], [200, 710], [1200, 710]])
    # Compute the transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse transformation matrix
    minv = cv2.getPerspectiveTransform(dst, src)
    # Apply the perspective transformation
    warped = cv2.warpPerspective(img, matrix, img_size)
    return warped, minv


def sliding_window_search(binary_warped):
    # Create a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    # Create an output image for visualization
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the midpoint of the histogram
    midpoint = int(histogram.shape[0] / 2)
    # Find the starting points for the left and right lanes
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Number of sliding windows
    nwindows = 9
    # Height of each window
    window_height = int(binary_warped.shape[0] / nwindows)
    # Get the positions of all non-zero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions for the sliding windows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Width of the windows
    margin = 100
    # Minimum number of pixels to recenter the window
    minpix = 50
    # Lists to store the indices of lane pixels
    left_lane_inds = []
    right_lane_inds = []

    # Loop through each sliding window
    for window in range(nwindows):
        # Define the boundaries of the windows
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Find non-zero pixels within the windows
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Add the indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter the windows if enough pixels are found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Combine all the indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Get the pixel positions of the lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If no lane pixels are found, return None
    if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
        return None, None, None, None, None

    # Fit a second-order polynomial to the lane lines
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Calculate x values for the lane lines
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return ploty, left_fit, right_fit, left_fitx, right_fitx


def measure_curvature(ploty, left_fitx, right_fitx):
    # Evaluate the curvature at the bottom of the image
    y_eval = np.max(ploty)
    # Convert pixel values to meters
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the radius of curvature for both lanes
    left_curverad = (
        (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
    ) / np.abs(2 * left_fit_cr[0])
    right_curverad = (
        (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
    ) / np.abs(2 * right_fit_cr[0])
    # Average the curvature of both lanes
    curvature = (left_curverad + right_curverad) / 2

    # Determine the direction of the curve
    if left_fit_cr[0] > 0 and right_fit_cr[0] > 0:
        curve_direction = "Right Curve"
    elif left_fit_cr[0] < 0 and right_fit_cr[0] < 0:
        curve_direction = "Left Curve"
    else:
        curve_direction = "Straight"

    return curvature, curve_direction


def draw_lane_lines(original_image, binary_warped, left_fitx, right_fitx, ploty, Minv):
    # Create a blank image to draw the lane on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Create points for the left and right lane lines
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill the area between the lane lines
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the lane area back to the original perspective
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (original_image.shape[1], original_image.shape[0])
    )
    # Overlay the lane area on the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result


def off_center(meanPts, inpFrame):
    # Calculate the deviation from the center of the lane
    mpts = meanPts[-1][-1][-2].astype(int)
    pixelDeviation = inpFrame.shape[1] / 2 - mpts
    deviation = pixelDeviation * xm_per_pix
    direction = "right" if deviation < 0 else "left"
    return deviation, direction


def add_text(img, radius, direction, deviation, devDirection):
    # Add curvature and direction information to the image
    font = cv2.FONT_HERSHEY_TRIPLEX
    if direction != "Straight":
        text = "Radius of Curvature: " + "{:04.0f}".format(radius) + "m"
        text1 = "Curve Direction: " + (direction)
    else:
        text = "Radius of Curvature: " + "N/A"
        text1 = "Curve Direction: " + (direction)

    cv2.putText(img, text, (50, 100), font, 0.8, (0, 100, 200), 2, cv2.LINE_AA)
    cv2.putText(img, text1, (50, 150), font, 0.8, (0, 100, 200), 2, cv2.LINE_AA)

    # Add off-center information to the image
    deviation_text = (
        "Off Center: " + str(round(abs(deviation), 3)) + "m" + " to the " + devDirection
    )
    cv2.putText(
        img, deviation_text, (50, 200), font, 0.8, (0, 100, 200), 2, cv2.LINE_AA
    )

    return img


while cap.isOpened():
    # Read a frame from the video or camera
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to detect edges
    processed_frame = process_image(frame)
    # Transform the frame to a bird's eye view
    warped, Minv = perspective_warp(processed_frame)
    # Detect lane lines using sliding windows
    ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window_search(warped)

    # If no lanes are detected, display a message
    if (
        ploty is None
        or left_fit is None
        or right_fit is None
        or left_fitx is None
        or right_fitx is None
    ):
        cv2.putText(
            frame,
            "No lanes detected",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Calculate curvature and direction
    curvature, curve_direction = measure_curvature(ploty, left_fitx, right_fitx)
    # Draw the detected lane on the original frame
    lane_frame = draw_lane_lines(frame, warped, left_fitx, right_fitx, ploty, Minv)

    # Calculate the deviation from the center
    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    deviation, directionDev = off_center(pts_mean, frame)

    # Add text information to the frame
    final_img = add_text(lane_frame, curvature, curve_direction, deviation, directionDev)

    # Display the final frame
    cv2.imshow("Lane Detection", final_img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()