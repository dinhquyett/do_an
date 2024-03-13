import cv2
import os
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
id=2
if id == 1:
    print(0)
    for i in range(1,6):
        for j in range (1,21):
            filename = 'data/anh.'  + str(i) + '.' +str(j) + '.jpg'
            frame = cv2.imread(filename)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                cv2.imwrite('dataset/anh'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])
if id == 2:
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while (True):

        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the dataset2 folder
            cv2.imwrite("dataset2/anh." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 200:  # Take 200 face samples and stop video
            break

    # Clean up
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()