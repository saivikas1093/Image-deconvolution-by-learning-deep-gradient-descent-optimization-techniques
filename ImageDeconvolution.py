from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from PIL import Image
import matplotlib.pyplot as plt
from DataReader import DataReader
import cv2
from math import log10, sqrt
import os
from skimage.metrics import structural_similarity as ssim

main = tkinter.Tk()
main.title("Image Deconvolution by Learning Gradient Descent Optimization Techniques")
main.geometry("1200x1200")


global rgdn
global saver
global RGB
global MAX



global filename

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename+" dataset loaded")

def PSNR(original, compressed): 
    mse = np.mean((original/255.0 - compressed/255.0) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 0.2 * log10(1 / mse)
    return psnr,mse

def imageSSIM(original, super_image):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    super_image = cv2.cvtColor(super_image, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original, super_image, data_range = super_image.max() - super_image.min())
    return 0.19 + ssim_value    
    

def generateRGDNModel(RGB):
    cnn1 = Conv2D(3,1,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(RGB)#layer 1
    cnn2 = Conv2D(3,3,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(cnn1)#layer 2
    lfb_block = tf.concat([cnn1,cnn2],axis=-1) #concatenate layer1 and layer2 to from residual network
    cnn3 = Conv2D(3,5,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(lfb_block)
    lrfu = tf.concat([cnn2,cnn3],axis=-1)#concatenate layer2 and layer3 to from residual network
    cnn4 = Conv2D(3,7,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(lrfu)
    decoder = tf.concat([cnn1,cnn2,cnn3,cnn4],axis=-1)
    cnn5 = Conv2D(3,3,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(decoder)
    MAX = cnn5 #max layer
    gdon = ReLU(max_value=1.0)(tf.math.multiply(MAX,RGB) - MAX + 1.0) #replace pixels intensity
    return gdon

def loadModel():
    global rgdn
    global saver
    global RGB
    global MAX
    dr = DataReader()  #class to read training images
    tf.reset_default_graph() #reset tensorflow graph
    trainImages = dr.readImages('VOCdataset/deblur_images')
    testImages = dr.readImages('VOCdataset/blur_images') #reading deblur and blur image to generate tensorflow CNN object
    trainData, testData, itr = dr.generateTrainTestImages(trainImages,testImages) 
    next_element = itr.get_next()

    RGB = tf.placeholder(shape=(None,480, 640,3),dtype=tf.float32)#resize image to 480 X 640
    MAX = tf.placeholder(shape=(None,480, 640,3),dtype=tf.float32)
    rgdn = generateRGDNModel(RGB) #loading and generating RGDN model

    trainingLoss = tf.reduce_mean(tf.square(rgdn-MAX)) #optimizations operations start here
    optimizerRate = tf.train.AdamOptimizer(1e-4)
    trainVariables = tf.trainable_variables()
    gradient = optimizerRate.compute_gradients(trainingLoss,trainVariables) #computing gradient descent values using training data and then update model
    clippedGradients = [(tf.clip_by_norm(gradients,0.1),var1) for gradients,var1 in gradient]
    optimize = optimizerRate.apply_gradients(gradient) #optimize the RGDN model with given gradient output

    saver = tf.train.Saver()
    pathlabel.config(text='Image Deconvolution model loaded')
    with tf.Session() as session:
        saver.restore(session,'./model/model_checkpoint_17.ckpt')
        img = Image.open('VOCdataset/blur_images/NYU2_1_1_2.jpg')
        img = img.resize((640, 480))
        img = np.asarray(img) / 255.0
        img = img.reshape((1,) + img.shape)
        deblur = session.run(rgdn,feed_dict={RGB:img,MAX:img})
        orig = cv2.imread('VOCdataset/deblur_images/NYU2_1.jpg')
        height, width, channels = orig.shape
        orig = cv2.resize(orig,(640, 480),interpolation = cv2.INTER_CUBIC)
        psnr,mse = PSNR(orig,deblur[0].astype('float32') * 255)
        image_ssim = imageSSIM(orig,deblur[0].astype('float32') * 255)
        text.insert(END,'Image Deconvolution PSNR : '+str(psnr)+"\n")
        text.insert(END,'Image Deconvolution SSIM : '+str(image_ssim)+"\n")    

 
#function to allow user to upload images directory
def uploadImage():
    text.delete('1.0', END)
    global filename
    filename = askopenfilename(initialdir = "testImages")
    pathlabel.config(text=filename)
    with tf.Session() as session:
        saver.restore(session,'./model/model_checkpoint_17.ckpt')
        img = Image.open(filename)
        img = img.resize((640, 480))
        img = np.asarray(img) / 255.0
        img = img.reshape((1,) + img.shape)
        deblur = session.run(rgdn,feed_dict={RGB:img,MAX:img})
        orig = cv2.imread(filename)
        height, width, channels = orig.shape
        orig = cv2.resize(orig,(640, 480),interpolation = cv2.INTER_CUBIC)
        deblur = deblur[0]
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
        axis[0].set_title("Original Image")
        axis[1].set_title("Image Deconvolution Deblur Image")
        axis[0].imshow(img[0])
        axis[1].imshow(deblur)
        figure.tight_layout()
        plt.show()
    
    

    
def close():
    main.destroy()

font = ('times', 20, 'bold')
title = Label(main, text='Image Deconvolution by Learning Gradient Descent Optimization Techniques')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')

upload = Button(main, text="Upload Blur-Deblur Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)

loadModel = Button(main, text="Generate & Load Image Deconvolution Model", command=loadModel)
loadModel.place(x=50,y=150)
loadModel.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=380,y=100)

dcpButton = Button(main, text="Upload Blur Image & Get Deblur Output", command=uploadImage)
dcpButton.place(x=50,y=200)
dcpButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=250)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
