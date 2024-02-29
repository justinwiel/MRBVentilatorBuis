import numpy as np
import cv2
import threading
from gpiozero import PWMOutputDevice
from time import sleep
cap = cv2.VideoCapture(0)


class MovingAvgFilter:
    def __init__(self,sampleSize) -> None:
        self.values = []
        self.sampleSize = sampleSize
    def addSample(self,sample):
        self.values.append(sample)
        if len(self.values) > self.sampleSize:
            self.values.pop(0)  # remove first element if maximum sample size is reached. 
            
    def getAvg(self):
        return  np.average(self.values) if len(self.values) >0 else 0



class fanControler: 
    def __init__(self,pin:int,bActive_high:bool=False) -> None:
        
        self.fan = PWMOutputDevice(pin,active_high=bActive_high)
        self.fan.frequency = 10000
    def setValue(self,val):
        if val >100:
            val = 100
        elif val <0:
            val = 0
        self.fan.value = val/100# 1 is max so devide input by a hundred
    #   sleep(0.05)
    
    
class BallFinder:
    def findOrange(self, frame):
        lower_orange = np.array([0, 204, 204])
        upper_orange = np.array([30, 255, 255])
        self.findColor(frame, lower_orange, upper_orange)

    def findColor(self, frame, lower, upper):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only orange colors
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours of the orange ball
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around the orange ball
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # Apply smoothing filter to contour points
                box_smoothed = cv2.convexHull(box)

                cv2.drawContours(frame, [box_smoothed], 0, (0, 255, 0), 2)
                print(f"found ball at {rect[0]} of size of width {rect[1][0]} and height {rect[1][1]}")
                
            
            
def sendPwm(fan,val):
  if val >100:
      val = 100
  elif val <0:
      val = 0
  fan.value = val/100# 1 is max so devide input by a hundred
#   sleep(0.05)

def main():
    control = fanControler(pin=14)
    find = BallFinder()
    
    while True:
        ret, frame = cap.read()
        

        t1 = threading.Thread(target=find.findOrange,args=[frame])
        t1.start()
        t1.join()
        control.setValue(1)
        
        # Display the resulting frame
        cv2.imshow('Orange Ball Tracking', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






