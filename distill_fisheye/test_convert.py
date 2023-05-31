import cv2
import numpy as np
import time
import sys
# import tkinter as tk
import torch
from torch.nn import functional as F
# np.set_printoptions(threshold=sys.maxsize)

class fisheye(object):
    def __init__(self) -> None:
        self.img = None 
        self.fimg = None
        
        para = []
        
        self.pitch = -0.4
        self.yaw=0
        self.trans_x=1
        self.trans_y=1
        self.f0 = 700
        self.zoom = 0.5
        self.position_y = 1
        self.position_x = 1
        self.size_x = 256#360
        self.size_y = 256
        self.model=1

    def norm2fisheye(self,imgs,label=False):
        # self.img=img
        B,w,h,C = imgs.shape
        # print(img.shape)
        map_y = []
        map_x = []
        for i in range(B):
            mx,my=self.model1(size=imgs.shape,label=label)
            my = torch.tensor(my).float().to(imgs.device)
            mx = torch.tensor(mx).float().to(imgs.device)
            map_x.append(mx)
            map_y.append(my)
        
        # print(map_x[0])
        # print(map_x[1])
        map_x = torch.stack(map_x,dim=0)
        map_y = torch.stack(map_y,dim=0)
        # map_y = torch.tensor(map_y).float().to(img.device)
        # map_x = torch.tensor(map_x).float().to(img.device)
        grid = torch.stack((map_x/((w)/2)-1,map_y/((h)/2)-1),dim=3)#.unsqueeze(0)
        if label==False:
            outp = F.grid_sample(imgs.permute(0,3,1,2),grid=grid,mode='bilinear')
        else:
            outp = F.grid_sample(imgs.permute(0,3,1,2),grid=grid,mode='nearest')
        # outp = cv2.remap(img,map_x,map_y,cv2.INTER_LINEAR)
        
        return outp


    def model1(self,size=(1,512,512,3),label=False):
        # Orthographic
        _,h,w,C = size
        self.size_x = w/2#360
        self.size_y = h/2
        print(w)
        # w,h=size[1],size[0]
        # fc=500
        # fc = int(self.zoom*w/512*(2*self.f0))
        trans_x = self.trans_x
        trans_y = self.trans_y
        yaw = self.yaw
        pitch = self.pitch #pitch angle of fisheye camera
        f0 = w/512*self.f0 #f0 gets bigger, distortion gets smaller
        fc = int(self.zoom*w/np.sin(np.arctan(w/(2*f0))))
        # fc = self.fc #fisheye focal length
        rx = int(self.size_x) #image size
        ry = int(self.size_y)
        
        ##build the transform map
        u=np.linspace(0,2*rx,2*rx)
        v=np.linspace(0,2*ry,2*ry)
        udst,vdst = np.meshgrid(u,v)
        v,u = vdst-self.position_y*ry+fc*np.sin(pitch)+450*fc*(1-trans_y)/f0 ,\
            udst-self.position_x*rx+fc*np.sin(yaw)+800*fc*(1-trans_x)/f0 #get proxy
        
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
        # map_x = np.array(map_x,dtype=np.float32)
        # map_y = np.array(map_y,dtype=np.float32)
        map_y = torch.tensor(map_y).float()
        map_x = torch.tensor(map_x).float()
        #transform
        # if label==True:
        #     self.fimg = cv2.remap(img,map_x,map_y,cv2.INTER_NEAREST,borderValue=255)
        # else:self.fimg = cv2.remap(img,map_x,map_y,cv2.INTER_LINEAR)
        return map_x,map_y

    

if __name__ == "__main__":
    img = cv2.imread("b0.png")
    f = fisheye(img)
    f.adjust()