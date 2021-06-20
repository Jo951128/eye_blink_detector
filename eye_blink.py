import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

from playsound import playsound
import winsound



IMG_SIZE = (34, 26)
threshold_value_eye=0.4
count_frame= 0
Alarm_frame= 50

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#change your model 
model = load_model('models/2021_06_14_16_42_22.h5')
model.summary()


#######crop eye######
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect
#######crop eye#######

# main
cap = cv2.VideoCapture(0)


while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(600, 600), fx=0.5, fy=0.5)
  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  ## face detector 
  faces = detector(gray)

  img=cv2.flip(img,1) # Flip the image.
  img_h, img_w,_ =img.shape #image width, height

  for face in faces: 
    shapes = predictor(gray, face) # 68-point landmark detectors
    
   
    shapes = face_utils.shape_to_np(shapes) # shape to numpy
  
    sh=np.array(shapes) # numpy to array 
  
    #36-48 out of 68 points (eyes point)
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42]) 
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    #resize 
    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    
    #normalization 
    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    
    #predict eye 
    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

    # visualize (threshold_value_eye=0.4)
    state_l = 'O' if pred_l > threshold_value_eye else '-'
    state_r = 'O' if pred_r > threshold_value_eye else '-'




    l1=(int(img_w)-eye_rect_l[0],eye_rect_l[1]) # rectangle x1 , y1 (left) 
    l2=(int(img_w)-eye_rect_l[2],eye_rect_l[3]) # rectangle x2 , y2 (left)

    r1=(int(img_w)-eye_rect_r[0],eye_rect_r[1]) # rectangle x1 , y1 (right)
    r2=(int(img_w)-eye_rect_r[2],eye_rect_r[3]) # rectangle x2 , y2 (right)

    cv2.rectangle(img, pt1=l1, pt2=l2, color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=r1, pt2=r2, color=(255,255,255), thickness=2)

    # Alarm (Alarm_frame=50)  

    if (pred_l<threshold_value_eye and pred_r<threshold_value_eye): #  if eyes close
          count_frame=count_frame+1      
    else: # eyes open (= count_frame initialization)
          count_frame=0

    if count_frame>Alarm_frame: # Alarm run
          string_sign="warning"
          cv2.putText(img, string_sign, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
          winsound.Beep(2400,1000) #sound 
          winsound.SND_PURGE
         
          
    else: 
          string_sign="safe" 
          
          cv2.putText(img, string_sign, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
   
    # eye state 
    cv2.putText(img,state_l, (int(img_w)-eye_rect_l[0]-20,eye_rect_l[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img,state_r, (int(img_w)-eye_rect_r[0]-20,eye_rect_r[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Facial Outlines
    sh=sh[0:27] 
    face_n=len(sh)
    
    for i in range(face_n):
        cv2.circle(img,(int(img_w)-sh[i][0],sh[i][1]),1,(0,0,255),1)
 
  #show image 
  cv2.imshow('result', img)
  

  #break image 
  if cv2.waitKey(1) == ord('q'):
    break

    
