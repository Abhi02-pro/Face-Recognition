# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box
# 3. Flatten the largest face image and save(grayscale image) in a numpy array
# 4.Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# initializing video camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
face_data = []
dataset_path = './data/'

file_name = input("Enter the name of person : ")

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue        
    
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key = lambda f : f[2]*f[3])
    # sorting the faces to pick the largest face among the available ones
    # faces[-1] will give the last (largest face area) from the faces list
    
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,225,225), 2)
        
        # Extract (crop out the required face) : Region of interest
        # frame[y,x] - note this here y comes first
        
        offset = 20
        # we are storing the grayscale image
        # face_section = gray_frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section, (100,100))
        
        # store every 10th face
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))
        
        skip += 1
        cv2.imshow('Face Section', face_section)
    
    cv2.imshow('Frame', frame)
        
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed == ord('q')):
        break
    

  
# converting our face list array into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)
    
# save this data into file system
np.save(dataset_path+file_name+'.npy', face_data)
print("Data Saved Successfully at "+dataset_path+file_name+'.npy')
    
    
cap.release()
cv2.destroyAllWindows()
    
