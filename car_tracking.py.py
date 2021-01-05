#!/usr/bin/env python
# coding: utf-8

# In[144]:


import cv2
import numpy as np
import matplotlib.pyplot as plt 
classifier_file='C:/Users/Faster/Desktop/car_detect.xml'
pedestrain_file='C:/Users/Faster/Desktop/cascade_harr_classify.xml'

img_file='E:/car/car3.jpg'
video=cv2.VideoCapture('E:/car/Driving New York City.mp4')
car_tracker=cv2.CascadeClassifier(classifier_file)
ped_tracker=cv2.CascadeClassifier(pedestrain_file)


# In[145]:


while(video.isOpened()):         
    ret, frame = video.read()
    if ret==True:         
        gray_s = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cars=car_tracker.detectMultiScale(gray_s,1.4,2)
        peds=ped_tracker.detectMultiScale(gray_s,1.2,2)
        for (x,y,w,h) in peds:
            cv2.rectangle(frame,(x+1,y+2),(x+w,y+h),(0,0,255),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        for (x,y,w,h) in cars:
            cv2.rectangle(frame,(x+1,y+2),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        
        
            
        cv2.imshow("Detector_frame",frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break        
 
        
video.release()
cv2.destroyAllWindows()


# In[73]:


plt.imshow(img)


# In[74]:


img = cv2.imread(img_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
cv2.waitKey()


# In[75]:


car_tracker=cv2.CascadeClassifier(classifier_file)
cars=car_tracker.detectMultiScale(img_gray)
print(cars)


# In[76]:


for (x,y,w,h) in cars:
    cv2.rectangle(img_gray,(x,y),(x+w,y+h),(0,0,255),2)
    
plt.imshow(img_gray)


# In[ ]:




