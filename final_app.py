# from cProfile import label
from pickle import FRAME
from queue import Empty
# from turtle import shape
import streamlit as st
import cv2 #computer vision
import pandas as pd
import cvlib as cv
import numpy as np 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.model import l
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
# from statistics import mode

















#Model selector 
model_name = st.sidebar.selectbox(
    "Please select model first ",
    ("Best_yet", "Augmented_CNN", "Transfer_learning(NA)","Simple_CNN")
)

#radio for input method
with st.sidebar:
    input_method = st.radio(
        "Choose Input  method",
        ( "Upload an image","Take a Picture","Webcam")
    )

# st.text(model_name)
with st.sidebar:
    st.text("Webcam works on localhost only")



###################BL for models ##############################


#various model and their input config,label  can be defined 
if model_name=="Best_yet":
    model=model = tf.keras.models.load_model("emotion_modelgithubbbbb.hdf5")#best model
    input_shape=(64,64) # resolution
    clr=1 # 1 ~ grayscale 
    labels= ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] #dict 

if model_name=="Augmented_CNN":
    model=model = tf.keras.models.load_model("Augmented_CNN_100_EPOCH_64_64_GRAYSCALE.h5")
    input_shape=(48,48) # resolution
    clr=1 # 1 ~ grayscale 
    labels= ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] #dict 


if model_name=="Simple_CNN":
    model=model = tf.keras.models.load_model("CNN_50_EPOCH_64_64_GRAYSCALE.h5")
    input_shape=(48,48) # resolution
    clr=1 # 1 ~ grayscale 
    labels= ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] #dict 

if model_name=="Transfer_learning":
    model=model = tf.keras.models.load_model("TRANSFER_25_EPOCH_224_224_RGB.h5")
    input_shape=(224,224) # resolution
    clr=3 # 1 ~ grayscale 
    labels= ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] #dict 
    



#########################webcam BL###########################
if input_method =="Webcam":
    st.title("Live Webcam is currently unavaible")

if input_method =="Webcam_ERROR":   
    # st.text("webcam test")
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    if run==True:
        show_face=st.checkbox("Show Detected Face")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    ##webcam button
     
    while run:
        #read from cam 
        status, frame = camera.read()
        
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #croppin
        face, confidence = cv.detect_face(frame)
        for idx, f in enumerate(face):
            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
            # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])
            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue
            cropped_face=face_crop
            if clr==1:
                face_crop=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_crop = cv2.resize(face_crop, (input_shape))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            # total.append(idx)
            output_label = labels[idx]
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, output_label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 200, 0), 2)
            
            # st.text(label)
            live_feed=[frame]
            if show_face==True:
                live_feed.append(cropped_face)

            FRAME_WINDOW.image(live_feed)
    # if total is not None:
    # st.text("Webcam turned off")




#########################UPLOADED file ###################


if input_method=="Upload an image":
    
    
    st.text("Please upload image file with a face")

    uploaded_file = st.file_uploader("Choose a image file")
    
    
    if uploaded_file  is not None:
        # Convert the file to an opencv image.
        

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        opencv_image = cv2.imdecode(file_bytes, 1)
        # st.image(opencv_image,channels="BGR")
        sha=opencv_image.shape
        


        
        # st.text(opencv_image.shape)
        # st.image(opencv_image)
        
        img= cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)

        
        
        
        #face finder 
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Detect faces
        faces = face_cascade.detectMultiScale(opencv_image, 1.1, 4)
        # st.text(len(faces))
        if len(faces)>1:
            st.title("Multiple Faces detected Please try again !!!")
        # let's display face rectangle
        for (x, y, w, h) in faces:
            # st.text("ZZ")
            cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (0, 255, ), 2)
            faces = img[y:y + h, x:x + w]
            # cv2.imshow("face",faces)
            # cv2.imwrite('face.jpg', faces)
            # cv2.imwrite('detcted.jpg', opencv_image)
            # cv2.imshow('img', img)
        
       

        
        if len(faces) !=0 :
            resized = cv2.resize(faces,input_shape)
            
            
        else:resized = cv2.resize(img,input_shape)
        

        # st.text(resized.shape)
        st.image(opencv_image, channels="BGR")
        

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis,...]
        # print("HI")

        Genrate_pred = st.button("Generate Prediction")    
        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            # st.title("Predicted Label for the image is {}".format(map_dict[1]))
            st.title("Predicted Label for the image is {}".format(labels[prediction]))
            st.text(labels)




#####################Take a Picture######################



if input_method=="Take a Picture":

    uploaded_file=st.camera_input("Take a picture")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        opencv_image = cv2.imdecode(file_bytes, 1)
        # st.image(opencv_image,channels="BGR")
        sha=opencv_image.shape
        


        
        # st.text(opencv_image.shape)
        # st.image(opencv_image)
        
        img= cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)

        
        
        
        #face finder 
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Detect faces
        faces = face_cascade.detectMultiScale(opencv_image, 1.1, 4)
        # st.text(len(faces))
        if len(faces)>1:
            st.title("Multiple Faces detected please try another image !!!")
        # let's display face rectangle
        for (x, y, w, h) in faces:
            # st.text("ZZ")
            cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (0, 255, ), 2)
            faces = img[y:y + h, x:x + w]
            # cv2.imshow("face",faces)
            # cv2.imwrite('face.jpg', faces)
            # cv2.imwrite('detcted.jpg', opencv_image)
            # cv2.imshow('img', img)
        
        # st.text("gazab hui gyo")
        
        if len(faces) !=0 :
            resized = cv2.resize(faces,input_shape)
            
            
        else:resized = cv2.resize(img,input_shape)
        

        # st.text(resized.shape)
        st.image(opencv_image, channels="BGR")
        

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis,...]
        # print("HI")

        Genrate_pred = st.button("Generate Prediction")    
        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            # st.title("Predicted Label for the image is {}".format(map_dict[1]))
            st.title("Predicted Label for the image is {}".format(labels[prediction]))
            st.text(labels)




#EDA
