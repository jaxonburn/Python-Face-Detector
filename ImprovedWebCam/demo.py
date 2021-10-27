from imutils.video import WebcamVideoStream
from imutils.video import FPS
import cv2.cv2 as cv2

print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

trained_face_data = cv2.CascadeClassifier('../FaceDetector/haarcascade_frontalface_default.xml')
# loop over some frames...this time using the threaded stream
while True:
    frame = vs.read()

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    fps.update()

    # Q Key
    if key == 81 or key == 113:
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        break

vs.stop()
cv2.destroyAllWindows()




