# Test the classifier with the webcam
import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('classifier/cascade.xml')
img_num = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]
        cv2.imwrite(f'pos/face_{img_num}.jpg', face_crop)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img_num += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()