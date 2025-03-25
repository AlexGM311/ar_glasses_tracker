import time

import cv2
import sys


tracker_types = {"MOSSE": cv2.legacy.TrackerMOSSE, "BOOSTING": cv2.legacy.TrackerBoosting,
                 "KCF": cv2.legacy.TrackerKCF, "CSRT": cv2.legacy.TrackerCSRT}
tracker_class = tracker_types["MOSSE"].create

# initialize OpenCV's special multi-object tracker
trackers = cv2.legacy.MultiTracker.create()
FPS = 60
frame_delay = 1.0 / FPS
# if a video path was not supplied, grab the reference to the web cam
video = cv2.VideoCapture('test-1.mp4')
if not video.isOpened():
    print("Could not open video")
    sys.exit()
# loop over frames from the video stream
paused = False
hint_text = ""
while True:
    start_time = time.time()
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ok, frame = video.read()

    # check to see if we have reached the end of the stream
    if ok is None or frame is None:
        break

    # resize the frame (so we can process it faster)

    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    success, boxes = trackers.update(frame)

    # loop over the bounding boxes and draw then on the frame
    if success:
        for bbox in boxes:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frame, hint_text, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 192), 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    processing_time = time.time() - start_time
    wait_time = max(1, int((frame_delay - processing_time) * 1000))
    time.sleep(wait_time / 1000)

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        paused = True
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        # create a new object t racker for the bounding box and add it
        # to our multi-object tracker
        tracker = tracker_class()
        trackers.add(tracker, frame, box)

    if paused:
        hint_text = ""
        while key != 13:
            key = cv2.waitKey(1) & 0xFF
            if ord('\r') != key != 255:
                hint_text += chr(key)
        paused = False
    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break



# if we are using a webcam, release the pointer
video.release()

# close all windows
cv2.destroyAllWindows()
