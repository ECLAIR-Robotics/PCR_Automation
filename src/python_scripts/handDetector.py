import mediapipe as mp
import random
import math
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

def is_point_inside_polygon(polygon_points, test_point):
    """
    Determine if a point is inside a given polygon or not.
    Polygon is a list of (x, y) pairs, and test_point is a single (x, y) pair.
    """
    n = len(polygon_points)
    inside = False
    
    x, y = test_point
    p1x, p1y = polygon_points[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def calculate_centroid(vertices):
    x_sum = 0
    y_sum = 0
    for vertex in vertices:
        x_sum += vertex[0]
        y_sum += vertex[1]
    return x_sum / len(vertices), y_sum / len(vertices)

def sort_vertices_clockwise(vertices):
    centroid = calculate_centroid(vertices)
    
    def sort_key(vertex):
        return math.atan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
    
    sorted_vertices = sorted(vertices, key=sort_key, reverse=True)  # Use reverse=False for counter-clockwise
    return sorted_vertices

def calc_edge(frame, origin, vector):
    height = frame.shape[0]
    width = frame.shape[1]
    origin_height = origin[1]
    origin_width = origin[0]
    vector_width = vector[0]
    vector_height = -vector[1]
    
    scalar_x = 0
    scalar_y = 0
    if (vector_width < 0): #moving to bottom left
        scalar_x = origin_width / abs(vector_width)
    else:
        scalar_x = (width-origin_width) / abs(vector_width)
    if (vector_height < 0): #moving to bottom left
        scalar_y = origin_height / abs(vector_height)
    else:
        scalar_y = (height-origin_height) / abs(vector_height)    
    return vector * max(scalar_x, scalar_y)

     

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Define the fingers by their node numbers
        fingers = {
            'Thumb': [4, 3, 2],
            'Index': [8, 7, 6, 5],
            'Middle': [12, 11, 10, 9],
            'Ring': [16, 15, 14, 13],
            'Pinky': [20, 19, 18, 17]
        }

        points = []
        for _ in range(200):
            x = random.randint(0, frame.shape[1])
            y = random.randint(0, frame.shape[0])
            points.append((x, y))
        
        #array to be filled with the approximate center of the hand
        hand_centers = []
        
        # Rendering results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #nodes used to average 'center'
                palm_nodes = (0,1,2,5,9,13,17)
                selected_landmarks = np.array([(hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y) for i in palm_nodes])
                center_x,center_y = np.mean(selected_landmarks, axis=0)
                
                wrist_x, wrist_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                wrist_weight = 10  # Adjust the weight as needed. Moves center closer to the wrist 
                # Calculate the weighted average
                total_weight = wrist_weight + len(palm_nodes)
                center_x = (wrist_x * wrist_weight + center_x * len(palm_nodes)) / total_weight
                center_y = (wrist_y * wrist_weight + center_y * len(palm_nodes)) / total_weight

                hand_centers.append((center_x, center_y))
                center_image = (int(center_x * image.shape[1]), int(center_y * image.shape[0]))
                cv2.circle(image, center_image, 5, (0, 255, 0), -1)
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Retrieve the center for the current hand
                    center_x, center_y = hand_centers[hand_index]
                    center_image = (int(center_x * image.shape[1]), int(center_y * image.shape[0]))

                    # Initialize variables to find the furthest node
                    furthest_distance = 0
                    furthest_node = None
                    furthest_node_index = -1

                    cur_index = 0
                    # Iterate through all landmarks to find the furthest from the center
                    for landmark in hand_landmarks.landmark:
                        landmark_image = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                        distance = np.sqrt((landmark_image[0] - center_image[0]) ** 2 + (landmark_image[1] - center_image[1]) ** 2)

                        if distance > furthest_distance:
                            furthest_distance = distance
                            furthest_node = landmark_image
                            furthest_node_index = cur_index
                        cur_index += 1

                    # Draw a line from the center to the furthest node
                    
                    if furthest_node is not None:
                        cv2.line(image, center_image, furthest_node, (255, 0, 0), 2)

                    finger_name, finger_nodes = next(((f_name, f_nodes) for f_name, f_nodes in fingers.items() if furthest_node_index in f_nodes), (None, None))
                    
                    if finger_name is not None:
                         # Calculate the finger average
                        avg_x = np.mean([hand_landmarks.landmark[idx].x for idx in finger_nodes])
                        avg_y = np.mean([hand_landmarks.landmark[idx].y for idx in finger_nodes])
                        finger_average = (int(avg_x * image.shape[1]), int(avg_y * image.shape[0]))
                        
                        # Draw a line from the finger average to the furthest node
                        cv2.line(image, finger_average, furthest_node, (0, 255, 0), 2)
                        
                        direction = np.subtract(furthest_node, finger_average)
                        normalized_direction = direction / np.linalg.norm(direction)
                        perpindicular_line = np.array([-normalized_direction[1], normalized_direction[0]])

                        angle_degrees = 10 

                        cone_vectorA = np.add(normalized_direction * math.cos(math.radians(angle_degrees)), perpindicular_line * math.sin(math.radians(angle_degrees)))*1000
                        cone_vectorB = np.subtract(normalized_direction * math.cos(math.radians(angle_degrees)), perpindicular_line * math.sin(math.radians(angle_degrees)))*1000
                        height = frame.shape[0]
                        width = frame.shape[1]

                        extended_pointA = np.add(furthest_node, cone_vectorA).astype(int)  # make a point out by an arbitrary length
                        extended_pointB = np.add(furthest_node, cone_vectorB).astype(int)  # make a point out by an arbitrary length
                        sorted_points = sort_vertices_clockwise([furthest_node, extended_pointA, extended_pointB])
                        
                        for i in range(len(points)):
                            if (is_point_inside_polygon(sorted_points,points[i])):
                                cv2.circle(image, points[i], 5, (0, 255, 0), -1)
                            else:
                                cv2.circle(image, points[i], 5, (0, 0, 255), -1)

                        cv2.line(image, furthest_node, tuple(extended_pointA), (0, 0, 255), 2)
                        cv2.line(image, furthest_node, tuple(extended_pointB), (0, 0, 255), 2)


            

            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
            
        
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


print("program complete")
