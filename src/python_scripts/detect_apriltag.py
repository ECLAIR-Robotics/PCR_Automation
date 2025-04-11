import robotpy_apriltag
import cv2
import numpy as np
from typing import List, Tuple
import glob


class AprilTags:
    def __init__(self) -> None:
        self.WIDTH = 640
        self.HEIGHT = 480
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.cap = cv2.VideoCapture(0)
        self.tag = robotpy_apriltag.AprilTagDetector()
        self.tag.addFamily("tag36h11", 3)

    def getHomography(self):
        WIDTH_OUTPUT = 640
        HEIGHT_OUTPUT = 480

        while True:
            # Capture frame-by-frame
            ret, col_frame = self.cap.read()
            frame = cv2.cvtColor(col_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (640, 480))
            warped_img = frame
            warped_img2 = col_frame
            l: robotpy_apriltag.AprilTagDetection = self.tag.detect(frame)
            if l is not None:
                for i in l:
                    # print(i.getCenter())
                    point = i.getCenter()
                    x: int = int(point.x)
                    y: int = int(point.y)

                    points = [(0, 0), (0, 0), (0, 0), (0, 0)]

                    for j in range(len(points)):
                        points[j] = (int(i.getCorner(j).x), int(i.getCorner(j).y))
                    if (float(points[3][0] - points[2][0]) == 0) or (float(points[2][0] - points[1][0]) == 0) or (float(points[0][0] - points[1][0]) == 0) or (float(points[3][0] - points[0][0]) == 0):
                        continue
                    print("Points: ", points)
                    m1: float = float(points[3][1] - points[2][1]) / float(
                        points[3][0] - points[2][0]
                    )
                    m2: float = float(points[2][1] - points[1][1]) / float(
                        points[2][0] - points[1][0]
                    )
                    m3: float = float(points[0][1] - points[1][1]) / float(
                        points[0][0] - points[1][0]
                    )
                    m4: float = float(points[3][1] - points[0][1]) / float(
                        points[3][0] - points[0][0]
                    )

                    image_points = [
                        (WIDTH_OUTPUT, 0),
                        (WIDTH_OUTPUT, HEIGHT_OUTPUT),
                        (0, HEIGHT_OUTPUT),
                        (0, 0),
                    ]

                    # Calculating the scaled up points
                    xB: int = int((-1 * HEIGHT_OUTPUT) * m2 / float(m1 - m2))
                    yB: int = int(m2 * (xB - HEIGHT_OUTPUT)) + WIDTH_OUTPUT

                    xC: int = int(
                        (WIDTH_OUTPUT - (HEIGHT_OUTPUT * m2) + (HEIGHT_OUTPUT * m3))
                        / (m3 - m2)
                    )
                    yC: int = -1 * int(m3 * (480 - xC))

                    xD: int = int((-1 * WIDTH_OUTPUT) * m3 / (m4 - m3))
                    yD: int = int(m4 * xD)

                    xA: int = int(640 / (m4 - m1))
                    yA: int = int(m4 * xA)

                    scaledUpPoints: List[Tuple[int]] = [
                        (xA, yA),
                        (xB, yB),
                        (xC, yC),
                        (xD, yD),
                    ]
                    minValueX: int = min(scaledUpPoints, key=lambda point: point[0])[0]
                    minValueY: int = min(scaledUpPoints, key=lambda point: point[1])[1]
                    print("minValueX: ", minValueX)
                    print("minValueY: ", minValueY)

                    
                    
                    print("scaled up points: ", scaledUpPoints)
                    for i in range(len(scaledUpPoints)):
                        scaledUpPoints[i] = (
                            scaledUpPoints[i][0] + abs(minValueX),
                            scaledUpPoints[i][1] + abs(minValueY),
                        )

                    for j in range(4):
                        cv2.circle(frame, points[j], 2, (50, 0, 0), thickness=j)
                    """"
                    A3 ----------- B2
                    |             |
                    |             |
                    C0 ----------- D1
                    
                    """

                    # width = max(width_AB, width_CD)
                    width = WIDTH_OUTPUT

                    # height = max(height_BD, height_AC)
                    height = HEIGHT_OUTPUT
                    input_pts = np.float32([points[3], points[0], points[1], points[2]])
                    scaled_input_pts = np.float32([scaledUpPoints[3], scaledUpPoints[0], scaledUpPoints[1], scaledUpPoints[2]])
                    
                    
                    output_pts = np.float32(
                        [
                            [0, 0],
                            [0, height - 1],
                            [width - 1, height - 1],
                            [width - 1, 0],
                        ]
                    )
                    obj3d = np.zeros((4, 3), np.float32) 
                    a = [0, 285, 275, 285, 275]
                    b = [0, 125, 125, 135, 135]
                    for i in range(4):
                        obj3d[i] = [a[i], b[i], 0]
                    
                    # obj_points = np.float32([(285, 125, 0), (275, 125, 0), (285, 135, 0), (275, 135, 0)])
                    ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera([obj3d], input_pts, (640, 480), None, None)
                    print("Error in projection : \n", ret) 
                    print("\nCamera matrix : \n", camera_mat) 
                    print("\nDistortion coefficients : \n", distortion) 
                    print("\nRotation vector : \n", rotation_vecs) 
                    print("\nTranslation vector : \n", translation_vecs)
                    
                    M = cv2.getPerspectiveTransform(input_pts, output_pts)
                    M2 = cv2.getPerspectiveTransform(scaled_input_pts, output_pts)
                    
                    print("M2: ", M2)
                    
                    warped_img2 = cv2.warpPerspective(
                        col_frame, M2, (int(width), int(height)), flags=cv2.INTER_LINEAR
                    )
                    print("Input points: ", input_pts)
                    
                    point_test = np.float32([points[3][0], points[3][1], 0])
                    print("Point test: ", point_test)
                    test_mul =  np.multiply(M, point_test)
                    print("Test mul: ", test_mul)

                    # print("M: ", M)
                    warped_img = cv2.warpPerspective(
                        frame, M, (int(width), int(height)), flags=cv2.INTER_LINEAR
                    )

                    cv2.line(frame, (x, 0), (x, self.HEIGHT), (0, 255, 0), thickness=2)
                    cv2.line(frame, (0, y), (self.WIDTH, y), (0, 255, 0), thickness=2)
                    
                    cv2.line(frame, scaledUpPoints[0], scaledUpPoints[1], (0, 255, 0), thickness=2)
                    cv2.line(frame, scaledUpPoints[1], scaledUpPoints[2], (0, 255, 0), thickness=2)
                    cv2.line(frame, scaledUpPoints[2], scaledUpPoints[3], (0, 255, 0), thickness=2)
                    cv2.line(frame, scaledUpPoints[3], scaledUpPoints[0], (0, 255, 0), thickness=2)

            # Display the frame
            cv2.imshow("Camera Feed", frame)
            cv2.imshow("Flat image", warped_img)
            cv2.imshow("Scaled up image", warped_img2)
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the capture object and close the window
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    april_tag_detector = AprilTags()
    april_tag_detector.getHomography()
