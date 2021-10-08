import cv2, time, pandas
from datetime import datetime

staticBack = None
motionList = [None, None]
time = []
df = pandas.DataFrame(columns = ["Start", "End"])
video = cv2.VideoCapture(8)

while True:
    check, frame = video.read()
    motion = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if staticBack is None:
        