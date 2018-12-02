import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
def get_tr_data():
	data=[]
	label=[]
	for i in range(13):
		l=listdir("./dataset/{}".format(i))
		for t in l:
			if '.png' in t:
				path="./dataset/{}/{}".format(i,t)
				temp=cv2.imread(path,0)
				temp=cv2.resize(temp,(32,32))
				temp=np.reshape(temp,[32,32])
				data.append(temp)
				label.append(i)
	data=np.array(data)
	label=np.array(label)
	return data,label

def rm_sk(img):
	mon=cv2.moments(img)
	val=abs(mon['mu02'])
	if val<0.01:
		temp=img.copy()
		return temp
	else:
		sk=mon['mu11']/mon['mu02']
		temp=np.float32([[1,sk,-16*sk],[0,1,0]])
		img=cv2.warpAffine(img,temp,(32,32),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
		return img

def train():
	data,label=get_tr_data()
	rand=np.random.RandomState(13)
	s=rand.permutation(len(data))
	data=data[s]
	label=label[s]
	dsk=list(map(rm_sk,data))
	hg=HoG()
	desc=[]
	for temp in dsk:
		desc.append(hg.compute(temp))
	desc=np.squeeze(desc)
	ind=int(0.7*len(dsk))
	tr_data,te_data=np.split(dsk,[ind])
	tr_label,te_label=np.split(label,[ind])
	tr_h,te_h=np.split(desc,[ind])
	model=cv2.ml.SVM_create()
	model.setGamma(0.51)
	model.setC(12.5)
	model.setKernel(cv2.ml.SVM_RBF)
	model.setType(cv2.ml.SVM_C_SVC)
	model.train(tr_h,tr_label)
	pickle.dump('model.pkl')
	return model

def HoG():
	grad=True
	lev=64
	g_corr=1
	thres=0.2
	hist_norm=0
	aper=1
	bin_n=9
	win_sig=-1.
	cell=(10,10)
	str_b=(5,5)
	size_b=(10,10)
	win=(20,20)
	temp=cv2.HOGDescriptor(win,size_b,str_b,cell,bin_n,aper,win_sig,hist_norm,thres,g_corr,lev,grad)
	return temp

def predict(model,data):
	rg=cv2.cvtColor(data,cv2.COLOR_BG2GRAY)
	temp=[cv2.resize(rg,(32,32))]
	temp_dsk=list(map(dsk,temp))
	hg=HoG()
	desc=np.array([hg.compute(temp_dsk[0])])
	fin=np.reshape(desc,[-1,desc.shape[1]])
	res=model.predict(fin)[0].ravel()
	res=int(res[0])
	return res

