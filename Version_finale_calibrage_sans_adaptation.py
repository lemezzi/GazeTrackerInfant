import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

# Get the screen resolution
screen_width, screen_height = pyautogui.size()

# Create an empty white background covering the full screen
background = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255  # White background (255, 255, 255)

# Define circle parameters
edge_radius = 3  # Radius of the red circles in the corners

# Define indices for iris and eye borders
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

# Initialize variables for calibration
calibration_flag = 0
lu, lb, ru, rb = None, None, None, None
lu2, lb2, ru2, rb2 = None, None, None, None

# Initialize MediaPipe FaceMesh
with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as face_mesh:
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        # Display the image covering the full screen
        cv.namedWindow("Background with Circles", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("Background with Circles", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Background with Circles", background)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            left_iris_landmarks = mesh_points[LEFT_IRIS]
            right_iris_landmarks = mesh_points[RIGHT_IRIS]

            # Calculate the center points for the left and right iris
            center_right = np.mean(left_iris_landmarks, axis=0, dtype=np.int32)
            center_left = np.mean(right_iris_landmarks, axis=0, dtype=np.int32)

            # Draw a point at the center of the right iris
            cv.circle(frame, tuple(center_right), 1, (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, tuple(center_left), 1, (255, 0, 255), 1, cv.LINE_AA)

            # Draw a point at the left border and right border of the right eye
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 1, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 1, (0, 255, 255), -1, cv.LINE_AA)

            # Draw a point at the left border and right border of the left eye
            cv.circle(frame, mesh_points[L_H_RIGHT][-1], 1, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT][0], 1, (0, 255, 255), -1, cv.LINE_AA)

            

            # Draw four red circles in the corners for calibration
            corner_positions = [(edge_radius, edge_radius),  # Top-left corner
                                (screen_width - edge_radius, edge_radius),  # Top-right corner
                                (edge_radius, screen_height - edge_radius),  # Bottom-left corner
                                (screen_width - edge_radius, screen_height - edge_radius)]  # Bottom-right corner

            for position in corner_positions:
                cv.circle(background, position, edge_radius, (0, 0, 255), -1)  # Red circles in the corners

            if calibration_flag >= 5:
                distance_AD = np.linalg.norm(np.array(corner_positions[0]) - np.array(corner_positions[2]))
                distance_BC = np.linalg.norm(np.array(corner_positions[1]) - np.array(corner_positions[3]))
                distance_moy_y = min(distance_AD, distance_BC)

                distance_AB = np.linalg.norm(np.array(corner_positions[0]) - np.array(corner_positions[1]))
                distance_DC = np.linalg.norm(np.array(corner_positions[2]) - np.array(corner_positions[3]))
                distance_moy_x = min(distance_AB, distance_DC)
                # Calibration calculations for left eye
                distance_AD_pup = np.linalg.norm(lu - lb)
                distance_BC_pup = np.linalg.norm(ru - rb)
                distance_moy_pup_y = max(distance_AD_pup, distance_BC_pup)

                distance_AB_pup = np.linalg.norm(lu - ru)
                distance_DC_pup = np.linalg.norm(lb - rb)
                distance_moy_pup_x = max(distance_AB_pup, distance_DC_pup)

                fact_x = distance_moy_x / distance_moy_pup_x
                fact_y = distance_moy_y / distance_moy_pup_y

                # Calibration calculations for right eye
                distance_AD_pup2 = np.linalg.norm(lu2 - lb2)
                distance_BC_pup2 = np.linalg.norm(ru2 - rb2)
                distance_moy_pup_y2 = max(distance_AD_pup2, distance_BC_pup2)

                distance_AB_pup2 = np.linalg.norm(lu2 - ru2)
                distance_DC_pup2 = np.linalg.norm(lb2 - rb2)
                distance_moy_pup_x2 = max(distance_AB_pup2, distance_DC_pup2)

                fact_x2 = distance_moy_x / distance_moy_pup_x
                fact_y2 = distance_moy_y / distance_moy_pup_y

                # Project the point based on calibration
                projected_point = (

    int((min((center_right[0] - lu[0]) * fact_x , (center_left[0] - lu2[0]) * fact_x2)) ),
    int((min((center_right[1] - lu[1]) * fact_y ,(center_left[1] - lu2[1]) * fact_y2)))
)
                print(projected_point, fact_x, fact_y)

                # Draw a circle at the projected point
                cv.circle(background, projected_point, 50, (0, 0, 255), -1)

                # Check if the projected point is to the right or left
                if projected_point[0] > 970:
                    print("right")
                else:
                    print("left")

                bx1 = 0
                by1 = 0

        cv.imshow("img", frame)


        
        
        
        key = cv.waitKey(1)
        if key ==ord("a"):
            if not calibration_flag:
                lu = center_right
                lu2 = center_left
            elif calibration_flag == 1:
                lb = center_right
                lb2= center_left
            elif calibration_flag == 2:
                ru = center_right
                ru2= center_left
            elif calibration_flag == 3:
                rb = center_right
                rb2= center_left
                calibration_flag = 4  
            calibration_flag += 1
        #print(calibration_flag)
                
                
        if key ==ord("q"):
            break
        
cap.release()
cv.destroyAllWindows()