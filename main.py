import cv2
import os
import all_operations
import pandas as pd
import numpy as np
import time
from mtcnn import MTCNN
from datetime import datetime

vid = cv2.VideoCapture(0)
casc_path = os.getcwd() + "/haar/haarcascade_frontalface_default.xml"
student_csv_path = os.getcwd()+'/Student Embed/students.csv'
student_embed_path = os.getcwd()+'/Student Embed/student_embedding.npz'

faceCascade = cv2.CascadeClassifier(casc_path)
student_df = pd.read_csv(student_csv_path)
student_embed = np.load(student_embed_path)

embeds = student_embed['arr_0']
ids = student_embed['arr_1']

detector = MTCNN()

prev_frame_time = 0
new_frame_time = 0

while (True):
    ret, frame = vid.read()

    # rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # faces = detector.detect_faces(rgb_img)

    # for result in faces:
    #     x, y, w, h = result['box']
    #     x1, y1 = x + w, y + h
    #     cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = all_operations.normalize(face)
        face_embed = all_operations.embedding(face)
        face_embed = all_operations.l2_normalize(face_embed)

        min_dis = 100
        id = ''
        i = 0
        for i in range(embeds.shape[0]):
            dis = all_operations.findEuclideanDistance(embeds[i], face_embed)
            if dis < min_dis:
                min_dis = dis
                id = ids[i]
        
        if min_dis <= 1.049: 
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            cv2.putText(frame, f'{id}: {min_dis:.2f}, time: {current_time}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

vid.release()
cv2.destroyAllWindows()
