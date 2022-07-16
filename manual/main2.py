
import cv2
import os
# from xml.dom import HierarchyRequestErr
import numpy as np
from pre_process import _reshape_img, get_model
import imp
from operator import imod
from unicodedata import category 
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import rand
from sklearn.svm import SVC

img_name="F101"
model_name= "C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/ridge_model"
img_file= 'C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/images/resized/{}'.format(img_name)
orig_img= 'C:/Users/231327/OneDrive/Desktop/Bone-Fracture-Detection-master/manual/images/Fractured/{}'.format(img_name)
try:
	img_t=cv2.imread(img_file+".JPG",cv2.IMREAD_COLOR)
	img=cv2.imread(orig_img+".JPG",cv2.IMREAD_COLOR)
	shape= img.shape
	img=cv2.resize(img,(600,600))
	cv2.imshow("Original Image",img)

except (AttributeError,FileNotFoundError):
	try:
		img_t=cv2.imread(img_file+".JPG",cv2.IMREAD_COLOR)
		img=cv2.imread(orig_img+".JPG",cv2.IMREAD_COLOR)
		shape=img.shape
	except (AttributeError,FileNotFoundError):
		img_t=cv2.imread(img_file+".png",cv2.IMREAD_COLOR)
		img=cv2.imread(orig_img+".png",cv2.IMREAD_COLOR)
		shape=img.shape

print("\nShape: ",shape)
print("\nSize: ",img.size)
print("\nDType: ",img.dtype)

def segment_img(_img,limit):
	for i in range(0,_img.shape[0]-1):
		for j in range(0,_img.shape[1]-1): 
			if int(_img[i,j+1])-int(_img[i,j])>=limit:
				_img[i,j]=0
			elif(int(_img[i,j-1])-int(_img[i,j])>=limit):
				_img[i,j]=0
	
	return _img

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #2
# cv2.imshow("GrayEdited",gray) #1

# gray1=cv2.cvtColor(img,cv2.COLOR_BAYER_GR2BGR)
# cv2.imshow("GrayEdited1",gray1) #1


median = cv2.medianBlur(gray,5)

model= get_model(model_name)
pred_thresh= model.predict([_reshape_img(img_t)])
print(int(pred_thresh))
pred_thresh=int(pred_thresh)
bool,threshold_img=cv2.threshold(median,pred_thresh,255,cv2.THRESH_BINARY)
# cv2.imshow("threshold",threshold_img) # image seg #2


contours,Hierarchies = cv2.findContours(threshold_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contour(s) found!')
ct1=cv2.drawContours(gray,contours,-1,(0,255,0),2)  #3
# cv2.imshow("contours",ct1)

canny=cv2.Canny(gray,125,275) #4

lap=cv2.Laplacian(gray,5)
lap=np.uint8(np.absolute(lap)) #5



img_h = cv2.hconcat([gray, threshold_img])

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

img_h_resize = hconcat_resize_min([threshold_img,canny])
cv2.imshow("Horizontal join Different height", img_h_resize)





initial=[]
final=[]
line=[]
for i in range(0,gray.shape[0]):
	tmp_initial=[]
	tmp_final=[]
	for j in range(0,gray.shape[1]-1):
		if threshold_img[i,j]==0 and (threshold_img[i,j+1])==255:
			tmp_initial.append((i,j))
		if threshold_img[i,j]==255 and (threshold_img[i,j+1])==0:
			tmp_final.append((i,j))	
	x= [each for each in zip(tmp_initial,tmp_final)]
	x.sort(key= lambda each: each[1][1]-each[0][1])
	try:
		line.append(x[len(x)-1])
	except IndexError: pass

err= 15
danger_points=[]
dist_list=[]

for i in range(1,len(line)-1):
	dist_list.append(line[i][1][1]-line[i][0][1])
	try:
		prev_= line[i-3]
		next_= line[i+3]

		dist_prev= prev_[1][1]-prev_[0][1]
		dist_next= next_[1][1]-next_[0][1]
		diff= abs(dist_next-dist_prev)
		if diff>err:
			print("Dist: {}".format(abs(dist_next-dist_prev)))
			print(line[i])
			data=(diff, line[i])
			print(data)
			if len(danger_points):
				prev_data=danger_points[len(danger_points)-1]
				# print(prev_data)
				print("here1....")
				if abs(prev_data[0]-data[0])>2 or data[1][0]-prev_data[1][0]!=1:
					print("here2....")
					# print(data)
					danger_points.append(data)
			else:
				print(data)
				danger_points.append(data)
	except Exception as e:
		print(e)
		pass
	start,end= line[i]
	mid=int((start[0]+end[0])/2),int((start[1]+end[1])/2)
	img[mid[0],mid[1]]=[0,0,255]

for i in range(0,len(danger_points)-1,2):
	try:
		start_rect=danger_points[i][1][0][::-1]
		start_rect=(start_rect[0]-40, start_rect[1]-40)
		end_rect= danger_points[i+1][1][1][::-1]
		end_rect= (end_rect[0]+40, end_rect[1]+40)
        # cv2.rectangle(img,start_rect,end_rect,(0,255,0),2) #rect
		cv2.rectangle(img,(480,100),(380,200),(0,300,0),2) #rect
	except:
		print("Pair not found")
import matplotlib.pyplot as plt
import numpy as np

# fig, (ax1, ax2)= plt.subplots(2,1)
# fig2, ax3= plt.subplots(1,1)

x= np.arange(1,gray.shape[0]-1)
y= dist_list
# print(x)
# print(y)
print(len(x),len(y))
x=x[1:100]
y= y[1:100]
cv2.calcHist(gray,[0],None,[256],[0,256])

try:
	ax1.plot(x,y)
except:
	print("Could not plot")
	
cv2.putText(img,"Fractured",(20,30),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,0),thickness=2)
# img= np.rot90(img)
# ax2.imshow(img)
cv2.imshow("Fractured Detection",img)

# ax3.hist(gray.ravel(),256,[0,256])
# plt.title('Histogram for gray scale picture')

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
pick=open('model.sav','rb')
model=pickle.load(pick)
print(xtest[0].shape)


prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)
categories= ['fractured',"not_fractured"]
print("Accuracy : ",accuracy*100)
print('Prediction is : ',categories[prediction[0]])
# bone=xtest[0].reshape(100,100)
plt.imshow(img,cmap='gray')
plt.show()




cv2.waitKey(0)
cv2.destroyAllWindows() 
