from ast import Import
from turtle import shape
from PIL import Image
import cv2
import numpy as np
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report


WIDTH = 310/2
HEIGHT = 568/2

size = (int(WIDTH), int(HEIGHT))

def resize_and_save(img_name):
    main_image = Image.open("C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/images/Fractured/{}".format(img_name))
    x= main_image.resize(size, Image.NEAREST)
    x.save("C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/images/resized/{}".format(img_name))
    # print(main_image.show())
    print("done............................")

def _reshape_img(arr):
    flat_arr=[]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                flat_arr.append(arr[i][j][k])


    return flat_arr

def _create_data(train_img_list, label_list):

    inp_arr=[]
    for img in train_img_list:
        img= cv2.imread(img)
        inp_arr.append(_reshape_img(img))
    
    inp_arr= np.array(inp_arr)
    return inp_arr,np.array(label_list)
def train_and_save(train_img_list, label_list, model_name):
    try:
        with open(model_name,"rb") as file:
            model= pickle.load(file)
    except FileNotFoundError:
        in_arr, out_arr= _create_data(train_img_list, label_list)
        model= Ridge(alpha=0.01,tol=0.000001,max_iter=5000,random_state=43).fit(in_arr,out_arr)

        with open(model_name,"wb") as file:
            pickle.dump(model,file)    
    return model

def get_model(model_name):
    try:
        with open(model_name,"rb") as file:
            model= pickle.load(file)
            return model
    except FileNotFoundError:
        print("{} doesn't exist. Train and save a model first".format(model_name))
        sys.exit(0)

if __name__=="__main__":
    for each in range(1,102):
        try:
            resize_and_save("F{}.JPG".format(each))
        except IOError:
            try:
                resize_and_save("F{}.JPG".format(each))
            except IOError:
                resize_and_save("F{}.JPG".format(each))

    from train_label import train_label, test_label

    train_img_list=[]
    train_label_list=[]
    
    for key in train_label.keys():
        train_img_list.append("C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/images/resized/"+key+".jpg")
        train_label_list.append(train_label[key])
    
    test_img_list=[]
    test_label_list=[]
    for key in test_label.keys():
        test_img_list.append("C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/images/resized/"+key+".jpg")
        test_label_list.append(test_label[key])
    print("Training started...")
    svm_model=train_and_save(train_img_list,train_label_list, "ridge_model")
    print("Training finished...")

    train_in_arr, train_out_arr= _create_data(train_img_list,train_label_list)
    test_in_arr, test_out_arr= _create_data(test_img_list,test_label_list)

    print("Training set score: {:.2f}".format(svm_model.score(train_in_arr, train_out_arr)))
    print("Test set score: {:.2f}".format(svm_model.score(test_in_arr, test_out_arr)))
    
    # print(classification_report(train_in_arr,train_out_arr))

