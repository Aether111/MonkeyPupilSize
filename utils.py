import ultralytics
from ultralytics import YOLO
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.signal import savgol_filter

def get_model():
    return YOLO("best.pt")

def is_pupil_like(contour, image_center, image_size):
    # Calculate contour properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    # Define thresholds for size and shape
    MIN_AREA = 0.01 * image_size[0] * image_size[1]  # 1% of the bounding box area
    MAX_AREA = 0.4 * image_size[0] * image_size[1]  # 50% of the bounding box area
    MIN_CIRCULARITY = 0.3
    MAX_CIRCULARITY = 1.0

    # Get the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    contour_center = np.array([cx, cy])

    # Check if contour is close to the center of the image
    dist_to_image_center = np.linalg.norm(contour_center - image_center)

    # Check if contour is within acceptable bounds
    if (MIN_AREA <= area <= MAX_AREA and
        MIN_CIRCULARITY <= circularity <= MAX_CIRCULARITY and
        dist_to_image_center < max(image_size) / 4):  # Within the central half of the bounding box
        return True
    else:
        return False

def find_pupil_size(image, blur=True, morph_transform=True):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to smooth the image
    if blur:
        blur_size = 17#image.shape[1] // 10
        blur_size += (blur_size + 1) % 2
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        blurred = gray

    # Use adaptive thresholding to handle different lighting conditions
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological transformations to remove noise and smooth the shape
    if morph_transform:
        kernel_size = image.shape[1] // 20  # Kernel size relative to image size
        kernel_size += (kernel_size + 1) % 2  # Kernel size must be odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # To close small holes or dark regions within the pupil
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        #thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, (2,2))

        # To remove small bright spots on the pupil boundary
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_GRADIENT, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Image center and size for calculating valid contours
    image_center = np.array([gray.shape[1] // 2, gray.shape[0] // 2])
    image_size = gray.shape

    # Filter contours based on size, shape characteristics, and centrality
    valid_contours = [cnt for cnt in contours if is_pupil_like(cnt, image_center, image_size)]

    if valid_contours:
        # Assume the contour closest to the center that matches criteria is the pupil
        pupil_contour = min(valid_contours, key=lambda cnt: np.linalg.norm(np.mean(cnt, axis=0) - image_center))

        # Optionally draw the contours on the original image for visualization
        #cv2.drawContours(image, [pupil_contour], -1, (0, 255, 0), 2)

        # Show the image with the pupil contour (optional)
        #cv2_imshow(image)  # If you want to visualize the result

        return cv2.contourArea(pupil_contour)
    else:
        return None

def smooth_pupil_areas(pupil_areas, window_length=59, polyorder=2):
    # Ensure the window_length is odd and less than the size of pupil_areas
    if window_length % 2 == 0:
        window_length += 1  # Make it odd
    if window_length > len(pupil_areas):
        window_length = len(pupil_areas) // 2 * 2 + 1  # Make it the largest odd number less than the array length

    # Apply the Savitzky-Golay filter to smooth the array
    smooth_areas = savgol_filter(pupil_areas, window_length, polyorder)

    return list(smooth_areas)

def ewma(pupil_areas, alpha=0.3):
    """
    Computes the Exponential Weighted Moving Average of a sequence of numbers.

    :param pupil_areas: List of pupil areas with 'None' for missing values.
    :param alpha: Smoothing factor within [0,1], where larger alpha discounts older observations faster.
    :return: Smoothed sequence of pupil areas.
    """
    ewma_values = []
    last_value = None  # Holds the last non-None value for initialization

    for area in pupil_areas:
        if area is not None:
            if last_value is None:
                # If it's the first non-None value, initialize last_value with it
                last_value = area
            else:
                # Apply the EWMA formula
                last_value = alpha * area + (1 - alpha) * last_value
        # Append the calculated EWMA value if last_value is not None, else append None
        ewma_values.append(last_value if last_value is not None else None)

    return ewma_values

def process_video(video_path, model=None, output_path=None, save_video=False, display_frames=False, track_pupil_area=True):
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer if saving the video
    out = None
    if save_video and output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for MP4 output
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # List to store pupil areas
    pupil_areas = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert the frame to PIL Image to use with YOLO model
        pil_image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model.predict(source=pil_image, conf=0.5, verbose=False)  # Adjust as per your model's method
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()

        if len(boxes) > 2:
          print(len(pupil_areas))

        pupil_area = 0

        total = len(boxes)

        if len(boxes) > 0:
            for box in boxes:
              box_thing = tuple(box)
              eye_image = pil_image.crop(tuple(box))
              eye_image = np.array(eye_image)
              area = find_pupil_size(eye_image)
              if area is not None:
                pupil_area += area
              else:
                total -= 1

            if pupil_area == 0:
              pupil_area = None
            else:
              pupil_area /= total

            pupil_areas.append(pupil_area)  # Append pupil area to the list

            if save_video or display_frames:
                # Draw results on the frame
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"Pupil Area: {pupil_area}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
          pupil_areas.append(None)

        if save_video:
            out.write(frame)  # Write frame to output video

    # Release everything if job is finished
    cap.release()
    if save_video and out:
        out.release()


    pupil_areas = ewma(pupil_areas)

    pupil_areas = smooth_pupil_areas(pupil_areas, 25)

    return np.array(pupil_areas)


def remove_flat_and_nan_segments(arr, min_flat_length=5):
    """
    Removes segments from a numpy array where the same number is repeated consecutively
    for at least 'min_flat_length' times and any segments containing NaNs.

    Parameters:
    arr (numpy array): The input array.
    min_flat_length (int): The minimum length of flat segment to be removed.

    Returns:
    numpy array: A copy of 'arr' with flat and NaN-containing segments removed.
    """
    if len(arr) == 0:
        return arr

    # Calculate differences between consecutive elements
    diffs = np.diff(arr)

    # Identify where the differences are not zero (change points)
    change_points = np.where(diffs != 0)[0]

    # Add the start and end indices for completeness
    change_points = np.concatenate(([0], change_points + 1, [len(arr)]))

    # Create an array to hold the filtered segments
    filtered_segments = []

    kept_indices = []

    # Iterate over segments defined by change points
    for start, end in zip(change_points[:-1], change_points[1:]):
        segment = arr[start:end]
        segment_length = end - start

        # Check if the segment contains NaNs
        if np.isnan(segment).any():
            continue  # Skip the segment if it contains any NaNs

        # Check if the segment is flat and of a minimum length
        ######if segment_length >= min_flat_length and np.all(segment == segment[0]):
        ######    continue  # Skip the segment if it's flat and meets the length criterion

        # Add the segment to the list of segments to keep
        filtered_segments.append(segment)
        kept_indices.extend(list(range(start,end)))

    # Concatenate all segments that were kept
    if filtered_segments:
        return np.concatenate(filtered_segments), kept_indices
    else:
        return np.array([]), None  # Return an empty array if all segments are removed
    
def plot_pupil_area(pupil_areas, save_number, fps=30):
  time_stamps = np.array([f"{np.round(i,2):.2f}" for i in np.linspace(start=0,stop=len(pupil_areas),num=len(pupil_areas)*fps)])
  pupil_areas_cut, kept_indices = remove_flat_and_nan_segments(pupil_areas)
  pupil_areas_cut, kept_indices = pupil_areas_cut[:-1], kept_indices[:-1]
  avg = np.mean(pupil_areas_cut)
  std = np.std(pupil_areas_cut)
  interval = scipy.stats.norm.interval(confidence=0.8, loc=avg, scale=std)
  plt.figure(figsize=(20, 8))
  plt.plot(pupil_areas_cut, label='Pupil Area over Time')
  plt.xticks(ticks=np.arange(0,pupil_areas_cut.shape[0]),labels=time_stamps[kept_indices])
  plt.locator_params(axis='x', nbins=4*int(float(time_stamps[kept_indices][-1])))
  plt.xlabel('Time (s)')
  plt.ylabel('Pupil Area (pixels)')
  plt.title('Pupil Area Analysis over Time')
  plt.legend()
  plt.hlines(avg, xmin=0, xmax=pupil_areas_cut.shape[0],linestyles='dashed',colors='red')
  avg_line = np.repeat(avg,repeats=pupil_areas_cut.shape[0])
  l_bound, u_bound = np.repeat(interval[0],repeats=pupil_areas_cut.shape[0]), np.repeat(interval[1],repeats=pupil_areas_cut.shape[0])
  plt.fill_between(np.arange(0,avg_line.shape[0]),u_bound,l_bound,alpha=0.1,color='green')
  plt.savefig(f"output{save_number}.png")
  return plt.gcf()
