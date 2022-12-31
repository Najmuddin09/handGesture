from flask import Flask, render_template, Response
import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    # variables

    width, height = 1080, 720
    folderpath = "presentation"
    imageNumber = 0
    hs, ws = int(120 * 1), int(213 * 1)
    gestureThreshold = 300
    buttonPressed = False
    buttonCounter = 0
    buttonDelay = 30
    annotations = [[]]
    annotationNumber = 0
    annotationStart = False

    camera.set(3, width)
    camera.set(4, height)

    # Get the list of presentation images
    pathImages = sorted(os.listdir(folderpath), key=len)

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            # height,width, _ = imageCurrent.shape
            pathFullImage = os.path.join(folderpath, pathImages[imageNumber])
            imageCurrent = cv2.imread(pathFullImage)
            height, width, _ = imageCurrent.shape
            hands, frame = detector.findHands(frame)
            cv2.line(frame, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

            if hands and buttonPressed is False:
                hand = hands[0]
                fingers = detector.fingersUp(hand)
                cx, cy = hand['center']
                lmlist = hand['lmList']

                xVal = np.interp(lmlist[8][0], [width // 2, width], [0, width])
                yVal = np.interp(lmlist[8][1], [100, height - 100], [0, height])
                indexFinger = int(xVal), int(yVal)
                # print(fingers)

                if cy <= gestureThreshold:

                    # Gesture 1 - Left
                    if fingers == [1, 0, 0, 0, 0]:
                        print("left")

                        if imageNumber > 0:
                            buttonPressed = True
                            annotations = [[]]
                            annotationNumber = 0
                            annotationStart = False
                            imageNumber -= 1
                    # Gesture 2 - Right
                    if fingers == [0, 0, 0, 0, 1]:
                        print("right")

                        if imageNumber < len(pathImages) - 1:
                            buttonPressed = True
                            annotations = [[]]
                            annotationNumber = 0
                            annotationStart = False
                            imageNumber += 1

                # Gesture 3 - show pointer
                if fingers == [0, 1, 0, 0, 0]:
                    cv2.circle(imageCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED, )

                # Gesture  4- Draw pointer
                if fingers == [0, 1, 1, 0, 0]:
                    if annotationStart is False:
                        annotationStart = True
                        annotationNumber += 1
                        annotations.append([])
                    cv2.circle(imageCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                    annotations[annotationNumber].append(indexFinger)
                else:
                    annotationStart = False

                # Gesture 5 - Erase
                if fingers == [0, 1, 1, 1, 0]:
                    if annotations:
                        if annotationNumber >= 0:
                            annotations.pop()
                            annotationNumber -= 1
                            buttonPressed = True

            # ButtonPressed Iterations

            if buttonPressed:
                buttonCounter += 1

                if buttonCounter > buttonDelay:
                    buttonCounter = 0
                    buttonPressed = False

            for i in range(len(annotations)):
                for j in range(len(annotations[i])):
                    if j != 0:
                        cv2.line(imageCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 255), 12)

            # Adding webcam image on the slide
            # imageSmall = cv2.resize(frame, (ws, hs))
            # h, w, _ = imageCurrent.shape
            # imageCurrent[0:hs, w - ws:w] = imageSmall
            # cv2.imshow("Image", frame)
            # cv2.imshow("slides", imageCurrent)
            # key = cv2.waitKey(1)
            # if key == ord("q"):
            #     break

            # ret, buffer = cv2.imencode('.jpg', frame)
            ret, buffer = cv2.imencode('.jpg', imageCurrent)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames1():
    # variables

    width, height = 1080, 720

    gestureThreshold = 250

    camera.set(3, width)
    camera.set(4, height)

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            hands, frame = detector.findHands(frame)
            cv2.line(frame, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 5)

            ret, buffer = cv2.imencode('.jpg', frame)
            # ret, buffer = cv2.imencode('.jpg', imageCurrent)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames2():
    # variables


    folderpath = "rules"
    imageNumber = 0



    # Get the list of presentation images
    pathImages = sorted(os.listdir(folderpath), key=len)

    # Hand Detector
    # detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:

        pathFullImage = os.path.join(folderpath, pathImages[imageNumber])
        imageCurrent = cv2.imread(pathFullImage)


        ret, buffer = cv2.imencode('.jpg', imageCurrent)
        frame = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')






@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video1')
def video1():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

