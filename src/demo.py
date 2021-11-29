import math
import cv2
import time
import numpy as np
import os

import keras
from keras.layers import  *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from mtcnn import MTCNN

from keras.models import load_model

from google.colab.patches import cv2_imshow
from IPython.display import Image, display


model = load_model("HyperFAS/model/fas.h5")



def load_mtcnn_model(model_path):
    mtcnn = MTCNN(model_path)
    return mtcnn



def test_one(X):
    TEMP = X.copy()
    X = (cv2.resize(X,(224,224))-127.5)/127.5
    t = model.predict(np.array([X]))[0]
    time_end=time.time()
    return t


def test_camera(mtcnn,index = 0):

            store={}
            for imgPath in os.listdir('outputs'):
              frame=cv2.imread('outputs/'+imgPath)
              image = frame

              img_size = np.asarray(image.shape)[0:2]

              bounding_boxes, scores, landmarks = mtcnn.detect(image)

              nrof_faces = bounding_boxes.shape[0]
              if nrof_faces > 0:
                  for det, pts in zip(bounding_boxes, landmarks):

                      det = det.astype('int32')
                      #print("face confidence: %2.3f" % confidence)
                      det = np.squeeze(det)
                      y1 = int(np.maximum(det[0], 0))
                      x1 = int(np.maximum(det[1], 0))
                      y2 = int(np.minimum(det[2], img_size[1]-1))
                      x2 = int(np.minimum(det[3], img_size[0]-1))

                      w = x2-x1
                      h = y2-y1
                      _r = int(max(w,h)*0.6)
                      cx,cy = (x1+x2)//2, (y1+y2)//2

                      x1 = cx - _r 
                      y1 = cy - _r 

                      x1 = int(max(x1,0))
                      y1 = int(max(y1,0))

                      x2 = cx + _r 
                      y2 = cy + _r 

                      h,w,c =frame.shape
                      x2 = int(min(x2 ,w-2))
                      y2 = int(min(y2, h-2))

                      _frame = frame[y1:y2 , x1:x2]
                      score = test_one(_frame)

                      store[imgPath]=score[0]
            

            maxKey=-1
            maxVal=-1
            for key in store:
              if store[key]>maxVal:
                maxVal=store[key]
                maxKey=key
            
            ans='Most similar face is '+ maxKey + ' with a confidence value of '+str(maxVal)
            print(ans)
            

            

      
                      


    

if __name__ == '__main__':
    
    mtcnn = load_mtcnn_model("HyperFAS/model/mtcnn.pb")

    test_camera(mtcnn)





