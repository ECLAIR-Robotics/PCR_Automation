from .calibration import calibrate
import cv2
import numpy as np
import robotpy_apriltag
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import math
from collections import deque

def channel_processing(channel):
    # Adaptive thresholding, ot used right now
    # TODO: Add this to frame processing to boost accuracy
    channel = cv2.adaptiveThreshold(channel, 255, 
                                     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     thresholdType=cv2.THRESH_BINARY, 
                                     blockSize=55, 
                                     C=7)
    # Morphological operations to clean up noise
    channel = cv2.dilate(channel, None, iterations=1)
    channel = cv2.erode(channel, None, iterations=1)
    return channel

def calculate_angle(P1, P2, P3, P4):
    # Step 1: Calculate the vectors
    A = np.array([P2[0] - P1[0], P2[1] - P1[1]])  # Vector from P1 to P2
    B = np.array([P4[0] - P3[0], P4[1] - P3[1]])  # Vector from P3 to P4

    # Step 2: Calculate the dot product and the magnitudes of the vectors
    dot_product = np.dot(A, B)
    mag_A = np.linalg.norm(A)
    mag_B = np.linalg.norm(B)

    # Step 5: Calculate the angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product / (mag_A * mag_B))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def inter_centre_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two circle centers."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def colliding_circles(circles):
    for index1, circle1 in enumerate(circles):
        for circle2 in circles[index1 + 1:]:
            x1, y1, radius1 = circle1[:3]
            x2, y2, radius2 = circle2[:3]
            # Check for collision or containment
            if inter_centre_distance(x1, y1, x2, y2) < radius1 + radius2:
                return True
    return False

def filter_overlapping_circles(circles):
    """
    Remove overlapping circles, keeping the earlier one in the array.
    Args:
        circles: np.ndarray of circles, each represented as [x, y, radius].
    Returns:
        np.ndarray: Filtered array of non-overlapping circles.
    """
    filtered_circles = []
    
    for index, circle1 in enumerate(circles):
        x1, y1, radius1 = circle1[:3]
        is_overlapping = False
        
        # Check against already-added circles
        for circle2 in filtered_circles:
            x2, y2, radius2 = circle2[:3]
            if inter_centre_distance(x1, y1, x2, y2) < radius1 + radius2:
                is_overlapping = True
                break
        
        # Add the circle if it's not overlapping
        if not is_overlapping:
            filtered_circles.append(circle1)
    
    return np.array(filtered_circles, dtype=np.int32)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if point1 and point2:
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def initialize_camera(camera_number) -> cv2.VideoCapture:
    """Initialize the default webcam and return the capture object."""
    cap: cv2.VideoCapture = cv2.VideoCapture(camera_number)
    if not cap.isOpened():
        print("Camera not found or not accessible")
        exit()
    return cap

def initialize_detector_and_estimator(mtx, tagSize, tagFamily: str = "tag25h9"):
    """
    Initializes the AprilTag detector and pose estimator using the provided camera.

    This function calibrates the camera, sets up an AprilTag pose estimator with the 
    appropriate intrinsic parameters, and initializes an AprilTag detector with the 
    specified tag family.

    Parameters:
        cap (cv2.VideoCapture): OpenCV video capture object for the camera.
        tagFamily (str, optional): The tag family to detect (default: "tag25h9").

    Returns:
        Tuple[robotpy_apriltag.AprilTagDetector, robotpy_apriltag.AprilTagPoseEstimator]: 
        The initialized AprilTag detector and pose estimator.
    """
    
    # Configure the AprilTag pose estimator with camera intrinsic parameters
    config = robotpy_apriltag.AprilTagPoseEstimator.Config(
        # tagSize=0.0605,  # Size of the AprilTag in meters
        tagSize=tagSize,  # Size of the AprilTag in meters
        fx=mtx[0, 0],  # Camera's focal length in the x-direction (pixels)
        fy=mtx[1, 1],  # Camera's focal length in the y-direction (pixels)
        cx=mtx[0, 2],  # Camera's principal point X-coordinate (pixels)
        cy=mtx[1, 2]   # Camera's principal point Y-coordinate (pixels) [Fixed typo]
    )

    # Initialize the AprilTag detector
    detector = robotpy_apriltag.AprilTagDetector()
    
    # Add the specified tag family to the detector
    if not detector.addFamily(tagFamily):
        raise RuntimeError(f"Failed to add AprilTag family: {tagFamily}")

    # Create an AprilTag pose estimator with the configured parameters
    estimator = robotpy_apriltag.AprilTagPoseEstimator(config)

    return detector, estimator, mtx

def process_image(frame, detector):
    """Convert frame to grayscale and apply Gaussian blur for edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)
    results = detector.detect(gray)
    circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                1,
                30,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=100
            )
    return gray, blurred, edges, results, circles

def process_circles(circles, num_circles, x_positions, y_positions, r_positions, frame):
    if circles is not None:
        # Convert detected circles to integer values
        circles = np.round(circles[0, :]).astype("int")

        # Remove overlapping circles
        circles = filter_overlapping_circles(circles)

        # Keep only the specified number of circles and sort them by x-coordinate
        circles = circles[:num_circles]
        circles = sorted(circles, key=lambda x: x[0])

        # Store detected circle positions
        for index, (x, y, r) in enumerate(circles):
            if index >= num_circles:
                break
            x_positions[index].append(x)
            y_positions[index].append(y)
            r_positions[index].append(r)

        # Define colors for visualizing detected circles
        colors = [(255, 0, 0), (0, 255, 0)]  # Adjust if tracking more circles
        curr_averages = []

        for i in range(num_circles):
            if len(x_positions[i]) > 0:
                # Compute average position and radius over the stored history
                x_avg = int(sum(x_positions[i]) / len(x_positions[i]))
                y_avg = int(sum(y_positions[i]) / len(y_positions[i]))
                r_avg = int(sum(r_positions[i]) / len(r_positions[i]))

                # Draw the detected circle and its center
                cv2.circle(frame, (x_avg, y_avg), r_avg, colors[i], 4)
                cv2.rectangle(frame, (x_avg - 5, y_avg - 5), (x_avg + 5, y_avg + 5), (0, 128, 255), -1)

                curr_averages.append((x_avg, y_avg, r_avg))

        return curr_averages
    return []

def process_apriltag(apriltags, estimator, frame):
    """
    Processes AprilTags detected in the frame, estimates pose, and visualizes the tags with specific outlines.

    Args:
        apriltags (list): List of detected AprilTags.
        estimator (object): AprilTag pose estimator.
        frame (numpy array): Frame on which to draw the tag visualization.

    Returns:
        list: A list of 3 elements corresponding to tag_id 0, 1, 2.
              Each element is a tuple (center, pose) if detected, or None if not detected.
    """
    tag_data = [None, None, None]  # Index 0 = tag_id 0, Index 1 = tag_id 1, Index 2 = tag_id 2
    
    corner_colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]  # Red, Orange, Yellow, Green

    for tag in apriltags:
        tag_id = tag.getId()
        if tag_id in [0, 1, 2]:
            pose = estimator.estimate(tag)
            color = (255, 0, 0) if tag_id == 0 else (255, 0, 255)  # Blue for 0, Purple for 1 or 2

            # Get corners and convert to points
            corners = tag.getCorners((0.0,) * 8)
            corners_points = [
                (int(corners[i]), int(corners[i + 1])) for i in range(0, 8, 2)
            ]

            # Draw corners and edges
            for i, point in enumerate(corners_points):
                cv2.circle(frame, point, 5, corner_colors[i], -1)

            for i in range(4):
                cv2.line(frame, corners_points[i], corners_points[(i + 1) % 4], color, 2)

            # Compute center
            center = tuple(np.mean(corners_points, axis=0, dtype=int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            tag_data[tag_id] = (center, pose)

    return tag_data

def compute_real_world_coordinates(center1, pose1, center2, pose2, mtx, frame):
    """
    Computes real-world coordinates of detected circles based on the average of two AprilTag pose estimations.

    Args:
        center1 (tuple): Center of AprilTag 1.
        pose1 (object): Pose of AprilTag 1.
        center2 (tuple): Center of AprilTag 2.
        pose2 (object): Pose of AprilTag 2.
        mtx (numpy array): Camera matrix for intrinsic parameters.
        frame (numpy array): Frame on which to overlay distance annotations.

    Returns:
        list: Averaged real-world coordinates of the computed point.
    """
    def compute_single_world_coord(center, pose):
        x, y = center
        x_tag, y_tag, z_tag = pose.translation()
        roll, pitch, yaw = pose.rotation().x, pose.rotation().y, pose.rotation().z
        r = R.from_euler('xyz', [roll, pitch, yaw])
        R_tag = r.as_matrix()
        n = R_tag[:, 2]

        r_dir = np.array([(x - mtx[0, 2]) / mtx[0, 0], (y - mtx[1, 2]) / mtx[1, 1], 1])
        r_dir /= np.linalg.norm(r_dir)

        p_camera = np.array([0, 0, 0])
        t = np.dot(n, (np.array([x_tag, y_tag, z_tag]) - p_camera)) / np.dot(n, r_dir)
        return p_camera + t * r_dir

    p1 = compute_single_world_coord(center1, pose1)
    p2 = compute_single_world_coord(center2, pose2)
    avg_point = (p1 + p2) / 2

    distance = np.linalg.norm(p1 - p2)

    midpoint = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
    cv2.putText(frame, f"Dist: {100*distance:.3f} cm", midpoint,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(frame, center1, center2, (255, 0, 0), thickness=2)

    return avg_point, distance

def transform_point_relative_to_tag0(tag0_pose, tag0_center, target_point_world, mtx, frame):
    """
    Given a world point, compute dx, dy relative to tag0's facing direction.
    """
    # Vector from tag0 to the new point
    x0, y0 = tag0_center
    direction_vector = np.array([target_point_world[0] - x0, target_point_world[1] - y0], dtype=np.float64)
    distance = np.linalg.norm(direction_vector)

    # Get Tag 0's local +x direction
    r0 = R.from_euler('xyz', [tag0_pose.rotation().x, tag0_pose.rotation().y, tag0_pose.rotation().z])
    x_axis = r0.as_matrix()[:, 0]
    x_proj = np.array([x_axis[0], x_axis[1]], dtype=np.float64)
    x_proj /= np.linalg.norm(x_proj)

    # Get angle between x_proj and direction_vector
    direction_vector /= distance
    dot = np.dot(x_proj, direction_vector)
    det = x_proj[0]*direction_vector[1] - x_proj[1]*direction_vector[0]
    angle_rad = np.arctan2(det, dot)
    angle_deg = np.degrees(angle_rad)

    # Resolve into components in Tag 0 frame
    dx = distance * math.cos(angle_rad)
    dy = distance * math.sin(angle_rad)

    return dx, dy, angle_deg

def set_up_birds_eye(camera_number, tagSize):
    cap = initialize_camera(camera_number)
    _, mtx, _, _, _ = calibrate(cap)
    detector, estimator, mtx = initialize_detector_and_estimator(mtx, tagSize)

    return cap, detector, estimator, mtx

def get_x_and_y_for_tag(cap, detector, estimator, offset_distance_m, mtx, tagNumber, max_frames=10):
    frame_count = 0

    try:
        while frame_count < max_frames:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            _, _, _, apriltags, _ = process_image(frame, detector)
            center_and_pose = process_apriltag(apriltags, estimator, frame)

            if tagNumber == 1 and center_and_pose[0] and center_and_pose[1]:
                r1 = R.from_euler('xyz', [center_and_pose[1][1].rotation().x,
                                          center_and_pose[1][1].rotation().y,
                                          center_and_pose[1][1].rotation().z])
                y_axis_neg = -r1.as_matrix()[:, 1]
                translation1 = np.array(center_and_pose[1][1].translation())
                new_point_world = translation1 + offset_distance_m * y_axis_neg
                translation0 = np.array(center_and_pose[0][1].translation())
                dx2, dy2, angle = transform_point_relative_to_tag0(
                    center_and_pose[0][1],
                    translation0[:2],
                    new_point_world[:2],
                    mtx,
                    frame
                )
                return -dy2, -dx2

            if tagNumber == 2 and center_and_pose[0] and center_and_pose[2]:
                r2 = R.from_euler('xyz', [center_and_pose[2][1].rotation().x,
                                          center_and_pose[2][1].rotation().y,
                                          center_and_pose[2][1].rotation().z])
                y_axis_neg_2 = -r2.as_matrix()[:, 1]
                translation2 = np.array(center_and_pose[2][1].translation())
                new_point_world_2 = translation2 + offset_distance_m * y_axis_neg_2
                translation0 = np.array(center_and_pose[0][1].translation())
                dx3, dy3, angle2 = transform_point_relative_to_tag0(
                    center_and_pose[0][1],
                    translation0[:2],
                    new_point_world_2[:2],
                    mtx,
                    frame
                )
                return -dy3, -dx3

            cv2.imshow('Video Feed', frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
    
    return None, None

def main():
    tagSize = 0.061 # in meters
    offset_distance_cm = 7.4 # in centimeters
    camera_number = 0 

    cap = initialize_camera(camera_number)
    _, mtx, _, _, _ = calibrate(cap)
    detector, estimator, mtx = initialize_detector_and_estimator(mtx, tagSize)

    offset_distance_m = offset_distance_cm / 100  # Convert to meters

    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            _, _, _, apriltags, _ = process_image(frame, detector)

            # AprilTag processing
            center_and_pose = process_apriltag(apriltags, estimator, frame)

            # Calculate distances from the tag to detected circles
            if center_and_pose[0] and center_and_pose[1]:
                r1 = R.from_euler('xyz', [center_and_pose[1][1].rotation().x,
                              center_and_pose[1][1].rotation().y,
                              center_and_pose[1][1].rotation().z])
                y_axis_neg = -r1.as_matrix()[:, 1]  # -Y axis

                # Step 2: Move 7.14 cm from Tag 1 along -Y axis
                translation1 = np.array(center_and_pose[1][1].translation())  # Tag 1 position
                new_point_world = translation1 + offset_distance_m * y_axis_neg  

                # Step 3: Get Tag 0's translation
                translation0 = np.array(center_and_pose[0][1].translation())

                # Step 4: Compute vector from Tag 0 to new point
                _ = new_point_world - translation0
                dx2, dy2, angle = transform_point_relative_to_tag0(
                    center_and_pose[0][1],
                    translation0[:2],  # x, y from pose
                    new_point_world[:2],
                    mtx,
                    frame
                )

                print("\nPoint 7.14 cm at -90° from Tag 1:")
                print(f"  Relative to Tag 0:")
                print(f"    Forward (x): {-dy2 * 100:.3f} cm")
                print(f"    Lateral (y): {-dx2 * 100:.3f} cm")
                print(f"    Angle from facing: {angle:.2f} degrees")

            if center_and_pose[0] and center_and_pose[2]:
                r2 = R.from_euler('xyz', [center_and_pose[2][1].rotation().x,
                                        center_and_pose[2][1].rotation().y,
                                        center_and_pose[2][1].rotation().z])
                y_axis_neg_2 = -r2.as_matrix()[:, 1]  # -Y axis of Tag 2

                # Step 2: Move 7.14 cm from Tag 2 along -Y axis
                translation2 = np.array(center_and_pose[2][1].translation())  # Tag 2 position
                new_point_world_2 = translation2 + offset_distance_m * y_axis_neg_2  # 7.14 cm = 0.0714 m

                # Step 3: Get Tag 0's translation
                translation0 = np.array(center_and_pose[0][1].translation())

                # Step 4: Compute vector from Tag 0 to new point
                _ = new_point_world_2 - translation0
                dx3, dy3, angle2 = transform_point_relative_to_tag0(
                    center_and_pose[0][1],
                    translation0[:2],
                    new_point_world_2[:2],
                    mtx,
                    frame
                )

                print("\nPoint 7.14 cm at -90° from Tag 2:")
                print(f"  Relative to Tag 0:")
                print(f"    Forward (x): {-dy3 * 100:.3f} cm")
                print(f"    Lateral (y): {-dx3 * 100:.3f} cm")
                print(f"    Angle from facing: {angle2:.2f} degrees")

            cv2.imshow('Video Feed', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
