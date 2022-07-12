import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    
    ret,frame = cap.read()
    
    if ret == False:
        continue
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # detectMultiScale(video_frame, scaling_factor, n_neighbors)
    # the above function will return the list of tuples [(x,y,w,h),...]
    # x,y are coordinates of the rectange and w,h are width and height    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,225), 2)
        
    cv2.imshow("Video Frame", frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()