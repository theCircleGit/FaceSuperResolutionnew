#! /usr/bin/python3
# coding=utf-8

# import torch
import cv2
from .hubconf import custom
# import mediapipe as mp

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

model = custom("model/person_det/yolov5/yolov5s.pt")  # Load Custom yolov5 Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set Model Settings
model.conf = 0.80  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = [0]  # filter by person class


def yolov5_detect_person(img, label):
    # if(img.shape[0] < 300):
    #     img = cv2.resize(img, (0,0), fx=4, fy=4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC BGR to RGB x(640,1280,3)
    results = model(img_rgb, size=640)

    # # Apply pose estimation using mediapipe
    # results_pose = pose.process(img_rgb)
    # # print(results.pose_landmarks)
    # if results_pose.pose_landmarks:
    #     mpDraw.draw_landmarks(img, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
    #     for id, lm in enumerate(results_pose.pose_landmarks.landmark):
    #         h, w,c = img.shape
    #         # print(id, lm)
    #         cx, cy = int(lm.x*w), int(lm.y*h)
    #         cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

    # results.print()
    # results.show()
    df = results.pandas().xyxy[0]
    person_count = df[df['name'] == 'person'].shape[0]
    if person_count > 1:
        label = ''

    # labels : True -> YOLOv5 original label; '' -> no label ; str -> str label
    img_bgr = cv2.cvtColor(results.render(labels=label)[0], cv2.COLOR_RGB2BGR)
    # img_bgr = img

    # cv2.imwrite("test.jpg", img_bgr)

    return img_bgr


if __name__ == '__main__':
    img = cv2.imread('./test.jpg', cv2.IMREAD_COLOR)
    yolov5_detect_person(img, label='')