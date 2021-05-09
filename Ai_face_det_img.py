import cv2

frontal_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choosen image to detect face test and also conv into grayscale using 0
img = cv2.imread('face1.png')



grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#detection
face_coordinates = frontal_face_data.detectMultiScale(grayscaled_img)


#drew rectangle (first 4 cords of img)(secind 3 are color of rect) last digit is thickness of rectangle line
#((x,y) (x+w,y+h)) [x,y,width,height]

for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w, y+h),(225,0,0),2)

print(face_coordinates)


#show image state in termimal
cv2.imshow('Face detector',img)
cv2.waitKey ()


print("Code complete")
