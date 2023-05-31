import cv2
import numpy as np
import time
import sys
import tkinter as tk

# np.set_printoptions(threshold=sys.maxsize)

class fisheye(object):
    def __init__(self) -> None:
        # self.img = None 
        # self.fimg = None
        # self.map = None
        
        para = []
        
        self.pitch = -0.4
        self.yaw=0
        self.trans_x=1
        self.trans_y=1
        self.f0 = 700
        self.zoom = 0.5
        self.position_y = 1
        self.position_x = 1
        self.size_x = 256
        self.size_y = 256
        self.model=1

    def norm2fisheye(self,img,label=False):
        # self.img=img
        map_x,map_y = self.model1(img,size=img.shape,label=label)
        return map_x,map_y


    def model1(self,img,size=(512,512,3),label=False):
        # Orthographic
        w,h,C = size
        self.size_x,self.size_y = w/2,h/2
        # w,h=self.img.shape[1],self.img.shape[0]
        f0 = w/512*self.f0 #f0 gets bigger, distortion gets smaller
        fc = int(self.zoom*w/np.sin(np.arctan(w/(2*f0))))
        # img= self.img
        trans_x = self.trans_x
        trans_y = self.trans_y
        yaw = self.yaw
        pitch = self.pitch #pitch angle of fisheye camera
        # fc = self.fc #fisheye focal length
        rx = int(self.size_x) #image size
        ry = int(self.size_y)  
        ##build the transform map
        u=np.linspace(0,2*rx,2*rx)
        v=np.linspace(0,2*ry,2*ry)
        udst,vdst = np.meshgrid(u,v)
        v,u = vdst-self.position_y*ry+fc*np.sin(pitch)+450*fc*(1-trans_y)/f0 ,\
            udst-self.position_x*rx+fc*np.sin(yaw)+250*w/512*fc*(1-trans_x)/f0 #get proxy
        
        # rotate the fisheye sphere
        r1 = np.sqrt(fc**2-u**2)
        yc = r1*np.sin(np.arcsin(v/r1)-pitch)

        r1 = np.sqrt(fc**2-yc**2)
        filter2 = np.arcsin(u/r1)-yaw
        # filter2[filter2>np.pi/2]=None
        # filter2[filter2<-np.pi/2]=None
        xc = r1*np.sin(filter2) 

        # convert the proxy into raw image
        r = np.sqrt(xc**2+yc**2)
        r0 = f0*np.tan(np.arcsin(r/fc))
        p_theta = np.arctan2(yc,xc)
        x,y = r0*np.cos(p_theta),r0*np.sin(p_theta)

        map_x = x+w/2*trans_x
        map_y = y+h/2*trans_y
        map_y = np.array(map_y,dtype=np.float32)
        map_x = np.array(map_x,dtype=np.float32)
        # self.map_x = map_x
        # self.map_y = map_y
        #transform
        return map_x,map_y
        # if label==True:
        #     return #cv2.remap(img,map_x,map_y,cv2.INTER_NEAREST,borderValue=255)
        # else:return  #cv2.remap(img,map_x,map_y,cv2.INTER_LINEAR)


    

if __name__ == "__main__":
    img = cv2.imread("b0.png")
    f = fisheye(img)
    f.adjust()