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
    def __init__(self,fan:PWMOutputDevice) -> None:
        
        self.fan = fan
        self.fan.frequency = 10000
    def setValue(self,val):
        if val >100:
            val = 100
        elif val <0:
            val = 0
        self.fan.value = val/100# 1 is max so devide input by a hundred
    #   sleep(0.05)
    
    
class BallFinder:
    def findOrange(self, frame,result):
        lower_orange = np.array([0, 204, 204])
        upper_orange = np.array([30, 255, 255])
        self.findColor(frame, lower_orange, upper_orange,result)

    def findColor(self, frame, lower, upper,result):
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
            if area > 20:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # Apply smoothing filter to contour points
                box_smoothed = cv2.convexHull(box)

                cv2.drawContours(frame, [box_smoothed], 0, (0, 255, 0), 2)
                result[0]= rect[0][1]
                return#return like this, to provide the option to use a seperate thread
        result[0] = 0# if no ball is found return 0 
        return

                
            
            
def sendPwm(fan,val):
  if val >95:
      val = 95
  elif val <0:
      val = 0
  fan.value = val/100# 1 is max so devide input by a hundred
#   sleep(0.05)







def main():
    fan = PWMOutputDevice(14,active_high=True)
    control = fanControler(fan)
    find = BallFinder()
    filter = MovingAvgFilter(3)
    result = [0]
    control.setValue(100)
    sleep(0.2)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 1
    fontColor              = (255,0,0)
    thickness              = 1
    lineType               = 2
    while True:


        
        ret, frame = cap.read()
        
        t1 = threading.Thread(target=find.findOrange,args=[frame,result])
        t1.start()
        t1.join()
        filter.addSample(result[0])
        # control.setValue(100)
        # control.setValue(95)
        # sleep(1)
        control.setValue(95)
        sleep(0.3)
        # print(f"val {filter.getAvg() }")
        # Display the resulting frame
        cv2.putText(frame,f'pos {filter.getAvg()}', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        cv2.imshow('Orange Ball Tracking', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






