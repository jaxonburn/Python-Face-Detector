import cv2.cv2 as cv2
from imutils.video import FPS

# Pre Trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
cv2.CAP_PROP_FPS
fps = FPS().start()


while True:
    successful_frame_read, frame = webcam.read()

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

# Clean up
webcam.release()
cv2.destroyAllWindows()

# # Pre Trained Data
# trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# img = cv2.imread('frank.jpg', cv2.IMREAD_UNCHANGED)
#
# width = int(img.shape[1] * .5)
# height = int(img.shape[0] * .5)
# resized_img = cv2.resize(img, (width, height))
#
# grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#
# face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
#
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#
# cv2.imshow('Face Detector', resized_img)
# cv2.waitKey()

print('Done')
