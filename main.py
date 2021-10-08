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
        staticBack = gray
        continue
    diffFrame = cv2.absdiff(staticBack, gray)
    threshFrame = cv2.threshold(diffFrame, 30, 255, cv2.THRESH_BINARY)[1]
    threshFrame = cv2.dilate(threshFrame, None, iterations=2)
    cnts,_ = cv2.findContours(threshFrame, cv2.RETL_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 10000
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    motionList.append(motion)
    motionList = motionList[-2:]
    
    if motionList[-1] == 1 and motionList[-2] == 0:
        time.append(datetime.now())
    
    if motionList[-1] == 0 and motionList[-2] == 1:
        time.append(datetime.now())
        
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference Frame", diffFrame)
    cv2.imshow("Threshold Frame", threshFrame)
    cv2.imshow("Color Frame", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        if motion == 1:
            time.append(datetime.now())
        
        break
    
for i in range(0, len(time), 2):
    df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index=True)
    
df.to_csv("TimeOfMovements.csv")
video.release()
cv2.destroyAllWindows()
