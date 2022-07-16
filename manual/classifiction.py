import imp
from operator import imod
import os
from unicodedata import category 
import  numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import rand
from sklearn.svm import SVC
# dir = 'C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/category/'

# categories= ['fractured',"not_fractured"]
# data= []


# for category in categories:
#     path = os.path.join(dir,category)
#     label=categories.index(category)

#     for img in os.listdir(path):
#         imgpath=os.path.join(path,img)
#         bone_img=cv2.imread(imgpath,0)
#         try:
#             bone_img=cv2.resize(bone_img,(100,100))
#             image=np.array(bone_img).flatten()
#             data.append([image,label])
#         except Exception as e:
#             pass

# print(len(data))

# pick_in=open('data.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()

pick_in=open('data.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()

# random.shuffle(data)
features= []
labels=[]

for feature,label in data:
    features.append(feature)
    labels.append(label)


# cv2.imshow("rk",feature[0])
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.25)

# model=SVC(C=1,kernel='poly',gamma='auto')
# model.fit(xtrain,ytrain)
# print("train")

# pick=open('model.sav','wb')
# pickle.dump(model,pick)
# pick.close()
# img_name="89"
# img_file= 'C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/category/fractured/{}'.format(img_name)
# img=cv2.imread(img_file+".JPG",cv2.IMREAD_COLOR)
# cv2.imshow("rk",img)
img=features[0]
pick=open('model.sav','rb')
model=pickle.load(pick)
print(xtest[0].shape)
img=cv2.resize(img,(100,100))
img=np.array(img).flatten()
print(img.shape)

prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)
categories= ['fractured',"not_fractured"]
print("Accuracy : ",accuracy)
print('Prediction is : ',categories[prediction[0]])
bone=xtest[0].reshape(100,100)
plt.imshow(bone,cmap='gray')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows() 