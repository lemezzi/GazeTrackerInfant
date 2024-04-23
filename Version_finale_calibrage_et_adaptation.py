import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui
import matplotlib.pyplot as plt


# Get the screen resolution
screen_width, screen_height = pyautogui.size()

# Create an empty white background covering the full screen
background = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255  # White background (255, 255, 255)

# Define circle parameters
edge_radius = 3  # Radius of the red circles in the corners
calibration_flag = 0
lu, lb, ru, rb = None, None, None, None
lu2, lb2, ru2, rb2 = None, None, None, None
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
L_B_LEFT = [23]
R_H_LEFT = [362]
R_H_RIGHT = [263]
last_variation = None

def gaze_detection_with_calibration(frame):
    img_h, img_w, _ = frame.shape
    
    # Initialize MediaPipe FaceMesh
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.35,
                                          min_tracking_confidence=0.25) as face_mesh:
        
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            left_iris_landmarks = mesh_points[LEFT_IRIS]
            right_iris_landmarks = mesh_points[RIGHT_IRIS]

            # Calculate the center points for the left and right iris
            center_right = np.mean(left_iris_landmarks, axis=0, dtype=np.int32)
            center_left = np.mean(right_iris_landmarks, axis=0, dtype=np.int32)
            
            face_up_left_land = mesh_points[54]
            face_up_right_land = mesh_points[284]
            face_bottom_left_land = mesh_points[132]
            face_bottom_right_land = mesh_points[361]

            src_points = np.array([face_up_left_land, face_up_right_land, face_bottom_left_land, face_bottom_right_land],
                                  dtype=np.float32)

            right_eye_right_landmark = mesh_points[R_H_RIGHT][0]
            right_eye_left_landmark = mesh_points[R_H_LEFT][0]
            left_eye_right_landmark = mesh_points[L_H_RIGHT][0]
            left_eye_left_landmark = mesh_points[L_H_LEFT][0]

            return center_left, center_right, left_eye_right_landmark, left_eye_left_landmark, right_eye_right_landmark, right_eye_left_landmark, src_points


def rectify_and_show(image, src_points):
    wid= 500
    hei = 750
    # Define the destination points for perspective transformation
    dst_points = np.array([[0, 0], [wid, 0], [0, hei], [wid, hei]], dtype=np.float32)

    # Compute the perspective transformation matrix
    matrix = cv.getPerspectiveTransform(src_points, dst_points)

    # Perform perspective transformation
    rectified_image = cv.warpPerspective(image, matrix, (wid,hei))

    
    return (cv.cvtColor(rectified_image, cv.COLOR_BGR2RGB))

def find_and_rectify_eye(frame):
    # Initialize MediaPipe FaceMesh
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5) as face_mesh:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            

            face_up_left_land = mesh_points[54]
            face_up_right_land = mesh_points[284]
            face_bottom_left_land = mesh_points[58]
            face_bottom_right_land = mesh_points[288]

            src_points = np.array([face_up_left_land, face_up_right_land, face_bottom_left_land, face_bottom_right_land],
                                  dtype=np.float32)

            # Perform rectification
            rectified_eye = rectify_and_show(frame, src_points)
            return rectified_eye

    return None  # Return None if no face landmarks are detected

def find_and_rectify_eye2(frame):
    # Initialize MediaPipe FaceMesh
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv.imread('test.jpg')
    # Convert into grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        src_points = np.array([
            (x, y),  # Top-left corner of the face bounding box
            (x + w, y),  # Top-right corner of the face bounding box
            (x, y + h),  # Bottom-left corner of the face bounding box
            (x + w, y + h)  # Bottom-right corner of the face bounding box
        ], dtype=np.float32)
        rectified_eye = rectify_and_show(frame, src_points)
        return rectified_eye

            # Perform rectification

    return None  # Return None if no face landmarks are detected



# Example usage:
cap = cv.VideoCapture(0)
while True:
    ret, frame2 = cap.read()
    frame2 = cv.flip(frame2, 1)
    rgb_frame = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
    img_h, img_w = frame2.shape[:2]
    frame = find_and_rectify_eye(frame2)
    # Display the image covering the full screen
    cv.namedWindow("Background with Circles", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Background with Circles", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow("Background with Circles", background)
    if not ret:
        break
    
    # Draw four red circles in the corners for calibration
    corner_positions = [(edge_radius, edge_radius),  # Top-left corner
                        (screen_width - edge_radius, edge_radius),  # Top-right corner
                        (edge_radius, screen_height - edge_radius),  # Bottom-left corner
                        (screen_width - edge_radius, screen_height - edge_radius)]  # Bottom-right corner
    for position in corner_positions:
        cv.circle(background, position, edge_radius, (0, 0, 255), -1)
    for position in corner_positions:
        cv.circle(background, position, edge_radius, (0, 0, 255), -1)  # Red circles in the corners
    
    # Call the gaze_detection_with_calibration function
    results = gaze_detection_with_calibration(frame)
    
    if results:
        center_left, center_right, left_eye_right_landmark, left_eye_left_landmark, right_eye_right_landmark, right_eye_left_landmark, src_points = results
        
        # Draw a point at the center of the right iris
        cv.circle(frame, tuple(center_right), 3, (0, 0, 255), 1, cv.LINE_AA)
        cv.circle(frame, tuple(center_left), 3, (0, 0, 255), 1, cv.LINE_AA)

        # Draw a point at the left border and right border of the right eye
        cv.circle(frame, tuple(right_eye_right_landmark), 1, (255, 255, 255), -1, cv.LINE_AA)
        cv.circle(frame, tuple(right_eye_left_landmark), 1, (0, 255, 255), -1, cv.LINE_AA)

        # Draw a point at the left border and right border of the left eye
        cv.circle(frame, tuple(left_eye_right_landmark), 1, (255, 255, 255), -1, cv.LINE_AA)
        cv.circle(frame, tuple(left_eye_left_landmark), 1, (0, 255, 255), -1, cv.LINE_AA)
        if lu is not None:
            cv.circle(frame, tuple(lu), 3, (255, 0, 0), -1)  # Blue circle for lu
        if lb is not None:
            cv.circle(frame, tuple(lb), 3, (255, 0, 0), -1)  # Blue circle for lb
        if ru is not None:
            cv.circle(frame, tuple(ru), 3, (255, 0, 0), -1)  # Blue circle for ru
        if rb is not None:
            cv.circle(frame, tuple(rb), 3, (255, 0, 0), -1)  # Blue circle for rb

        # Draw lu2, lb2, ru2, rb2 in black
        if lu2 is not None:
            cv.circle(frame, tuple(lu2), 3, (0, 0, 0), -1)  # Black circle for lu2
        if lb2 is not None:
            cv.circle(frame, tuple(lb2), 3, (0, 0, 0), -1)  # Black circle for lb2
        if ru2 is not None:
            cv.circle(frame, tuple(ru2), 3, (0, 0, 0), -1)  # Black circle for ru2
        if rb2 is not None:
            cv.circle(frame, tuple(rb2), 3, (0, 0, 0), -1)
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
                #projected_point = (

    #int((min((center_right[0] - (lu[0] + dx) )  * fact_x , (center_left[0] - (lu2[0] + dx ) ) * fact_x2)) ),
    #int((min((center_right[1] - (lu[1] + dy) ) * fact_y ,(center_left[1] - (lu2[1] + dy) ) * fact_y2)))
#)
            projected_point = (

    int((min((center_right[0] - (lu[0] ) )  * fact_x , (center_left[0] - (lu2[0]  ) ) * fact_x2)) ),
    int((min((center_right[1] - (lu[1] ) ) * fact_y ,(center_left[1] - (lu2[1] ) ) * fact_y2))  )
)
            if projected_point[0] > 1920:
                projected_point = (1920, projected_point[1])
            if projected_point[1] > 1080:
                projected_point = (projected_point[0], 1080)
            if projected_point[0] < 0:
                projected_point = (0, projected_point[1])
            if projected_point[1] < 0:
               projected_point = (projected_point[0], 0)
            print(projected_point, fact_x, fact_y)

                # Draw a circle at the projected point
            cv.circle(background, projected_point, 50, (0, 0, 255), -1)

    cv.imshow("Eye", frame)
    key = cv.waitKey(1)
    if key ==ord("a"):
        if not calibration_flag:
            d=6
            lu = center_right
            lu2 = center_left
            lu[0]=lu[0]-d
            lu[1]=lu[1]-d
            lu2[0]=lu2[0]-d
            lu2[1]=lu2[1]-d
        elif calibration_flag == 1:
            lb = center_right
            lb2= center_left
            lb[0]=lb[0]-d
            lb[1]=lb[1]+d
            lb2[0]=lb2[0]-d
            lb2[1]=lb2[1]+d
        elif calibration_flag == 2:
            ru = center_right
            ru2= center_left
            ru[0]=ru[0]+d
            ru[1]=ru[1]-d
            ru2[0]=ru2[0]+d
            ru2[1]=ru2[1]-d
        elif calibration_flag == 3:
            rb = center_right
            rb2= center_left
            rb[0]=rb[0]+d
            rb[1]=rb[1]+d
            rb2[0]=rb2[0]+d
            rb2[1]=rb2[1]+d
            calibration_flag = 4  
            
            print(lu,lb,ru,rb,lu2,lb2,ru2,rb2)
        calibration_flag += 1
    
    if key ==ord("q"):
        break
cap.release()
cv.destroyAllWindows()
