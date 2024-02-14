import numpy as np
import cv2
import threading
from gpiozero import PWMLED
from time import sleep
cap = cv2.VideoCapture(0)
fan = PWMLED(14)


def findOrange(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range of orange color in HSV
    lower_orange = np.array([0 , 204 , 204 ])
    upper_orange = np.array([30 , 255, 255])
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
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
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            #print(f"found ball at {rect[0]} of size of width {rect[1][0]} and height {rect[1][1]}")
            
            
def sendPwm():
    try:
  # fade in and out forever
        while True:
            #fade in
            for duty_cycle in range(0, 100, 1):
            fan.value = duty_cycle/100.0
            sleep(0.05)

            #fade out
            for duty_cycle in range(100, 0, -1):
            fan.value = duty_cycle/100.0
            sleep(0.05)
      
    except KeyboardInterrupt:
        print("Stop the program and turning off the fan")
        fan.value = 0
        pass

while True:
    ret, frame = cap.read()
    

    t1 = threading.Thread(target=findOrange,args=[frame])
    t1.start()
    t1.join()
    sendPwm()
    
    # Display the resulting frame
    cv2.imshow('Orange Ball Tracking', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()








