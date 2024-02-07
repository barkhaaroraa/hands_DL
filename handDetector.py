import cv2
import mediapipe as mp
from time import time

# def detectPose(frame, pose_model, display=True):
#     modified_frame = frame.copy()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose_model.process(frame_rgb)
#     height, width, _ = frame.shape
#     landmarks = []
#     if results.pose_landmarks:
#         for landmark in results.pose_landmarks.landmark:
#             landmarks.append((landmark.x * width, landmark.y * height))
#         connections = mp.solutions.pose.POSE_CONNECTIONS
#         for connection in connections:
#             start_point = connection[0]
#             end_point = connection[1]
#             cv2.line(modified_frame, (int(landmarks[start_point][0]), int(landmarks[start_point][1])),
#                      (int(landmarks[end_point][0]), int(landmarks[end_point][1])), (0, 255, 0), 3)
#     else:
#         return None, None  
#     if display:
#         cv2.imshow('Pose Landmarks', modified_frame)
#     return modified_frame, landmarks

# def detect_fall(previous_height, previous_width, current_height, current_width, height_threshold=30):
#     if previous_height is None or previous_width is None:
#         return False  # Return false if any previous value is null

#     # Check if the change in height is significant
#     height_change = previous_height - current_height
#     if height_change > height_threshold:
#         return True  # Fall detected if height decreases significantly
#     return False

# pose_video = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
# video = cv2.VideoCapture(0)
# time1 = 0
# frame_count = 0
# previous_height = None
# previous_width = None

# while video.isOpened():
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Detect pose and landmarks
#     modified_frame, landmarks = detectPose(frame, pose_video, display=True)
#     frame_count += 1

#     # Calculate height and width every 20 frames
#     if frame_count % 20 == 0:
#         current_height = landmarks[0][1] - landmarks[11][1]  # Distance between eye and toe
#         current_width = abs(landmarks[12][0] - landmarks[11][0])  # Distance between hips
#         if detect_fall(previous_height, previous_width, current_height, current_width):
#             cv2.putText(modified_frame, 'Fall detected!', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
#         previous_height = current_height
#         previous_width = current_width

#     # Measure frames per second
#     time2 = time()
#     if (time2 - time1) > 0:
#         frames_per_second = 1.0 / (time2 - time1)
#         cv2.putText(modified_frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

#     time1 = time2

#     # Display the frame
#     cv2.imshow('Pose Detection', modified_frame)

#     # Check for exit key
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# # Release video capture and close OpenCV windows
# video.release()
# cv2.destroyAllWindows()



mphands=mp.solutions.hands
mpdraw=mp.solutions.drawing_utils

class handdetector:
    def __init__(self,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.hands=mphands.Hands(max_num_hands=max_num_hands,min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self,image,handnumber=0,draw=False):
        originalimg=image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result=self.hands.process(image)
        landMarkList = []
        if result.multi_hand_landmarks:
            hand=result.multi_hand_landmarks[handnumber]

            for id,landmar in enumerate(hand.landmark):
                imgh,imgw,imgc=originalimg.shape
                xPos,yPos=int(landmar.x *imgw), int(landmar.y*imgh)
                landMarkList.append([id,xPos,yPos])
            if draw:
                mpdraw.draw_landmarks(originalimg,hand,mphands.HAND_CONNECTIONS)
        return landMarkList
