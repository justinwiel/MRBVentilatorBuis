import numpy as np
import os
import cv2
import threading
from gpiozero import PWMOutputDevice, Device
from gpiozero.pins.mock import MockFactory, MockPWMPin
from time import sleep
import time as t
cap = cv2.VideoCapture(0)
time = 0
integral = 0
time_prev = -1e-6
e_prev = 0
allowedDiff = 0


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
        if val >99:
            val = 99
        elif val <0:
            val = 0
        # print(val)
        self.fan.value = val/100.0# 1 is max so devide input by a hundred
    #   sleep(0.05)
    
    
    
class BallFinder:
    def findOrange(self, frame,result):
        lower_orange = np.array([0, 200, 200])
        upper_orange = np.array([35, 255, 255])
        self.findColor(frame, lower_orange, upper_orange,result)

    def findColor(self, frame, lower, upper,result):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only orange colors
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours of the orange ball
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggestArea = 10
        # Draw bounding box around the orange ball
        result[0] = 100# if no ball is found return 100 as the height 
        selectedContour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > biggestArea:
                biggestArea = area
                selectedContour=contour
        rect = cv2.minAreaRect(selectedContour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Apply smoothing filter to contour points
        box_smoothed = cv2.convexHull(box)

        cv2.drawContours(frame, [box_smoothed], 0, (0, 255, 0), 2)
        result[0]= rect[0][1]
        return#return like this, to provide the option to use a seperate thread 

                
def PID(Kp, Ki, Kd, setpoint, measurement,offset = 0):
    global time, integral, time_prev, e_prev# Value of offset - when the error is equal zero
    # PID calculations
    e = setpoint - measurement
    if (e > 0 and e < allowedDiff ) or (e < 0 and e > -1*allowedDiff ): #wannneer de bal binnen de acceptabele waarden is, breng de waarde handmatig naar nul.q
        e = 0
    P = Kp*e
    integral = integral + Ki*e*(time - time_prev)
    D = Kd*(e - e_prev)/(time - time_prev)# calculate manipulated variable - MV
    MV = offset + P + integral + D
    # update stored data for next iteration
    e_prev = e
    time_prev = time
    return MV 
            

def on_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > 350:
            param[0] = 350
        elif y < 130:
            param[0] = 130
        else:
            param[0] = y



def main():
    global time
    # line below is for on desktop testing
    print(os.name)
    if os.name == 'nt':
        Device.pin_factory = MockFactory(pin_class=MockPWMPin)#aanwezig voor testen op PC irrelevant voor eindproduct
    fan = PWMOutputDevice(14,active_high=True)
    control = fanControler(fan)
    find = BallFinder()
    filter = MovingAvgFilter(1)
    result = [None]
    control.setValue(100)
    sleep(0.1)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 1
    fontColor              = (127,255,0)
    thickness              = 1
    lineType               = 2
    setpointList = [300]
    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame,0)
        # if measurement==480:
        #     measurement = 0
        
        t1 = threading.Thread(target=find.findOrange,args=[frame,result])
        t1.start()
        t1.join()
        filter.addSample(result[0])
        measurement = filter.getAvg()
        # control.setValue(100)
        # control.setValue(95)
        # sleep(1)
        P,I,D = .035,.0000000000_030,.065
        setpoint = setpointList[0]
        offset = 72.85
        PID_res = PID(P,I,D,setpoint,measurement,offset)
        if(setpoint <= 90):
            PID_res = 0

        # print(PID_res)
        control.setValue(PID_res)
        time = t.time()
        # sleep(.1)
        # print(f"val {filter.getAvg() }")
        # Display the resulting frame
        cv2.putText(frame,f'pos {measurement}', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        cv2.putText(frame,f'PID {PID_res}', 
        (bottomLeftCornerOfText[0],bottomLeftCornerOfText[1]+25), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        cv2.putText(frame,f'setpoint {setpoint}', 
        (bottomLeftCornerOfText[0],bottomLeftCornerOfText[1]+50), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        cv2.namedWindow('PID window')
        cv2.setMouseCallback('PID window',on_click,param=setpointList)
        frame = cv2.line(frame, (0,setpointList[0]), (800,setpointList[0]), fontColor, 2)
        cv2.imshow('PID window', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print(fan.value)

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






