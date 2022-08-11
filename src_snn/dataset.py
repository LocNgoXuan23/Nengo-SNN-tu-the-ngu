import json
import os
import numpy as np
from scipy import ndimage
import tensorflow as tf
import cv2
from utils import *
from torch.utils.data import Dataset, DataLoader
import random 
import torch
from torchvision import  transforms
from tqdm import tqdm

def get_data_cnn(path):
	data = read_json(path)
	imgs = []
	labels = []

	for d in tqdm(data):
		label = get_label(d)
		img = cv2.imread(d)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# img = img.reshape(224,224)
		img = cv2.resize(img, (224, 224))
		labels.append(label)
		imgs.append(img)
	
	imgs = np.array(imgs)
	labels = np.array(labels)
	
	print(imgs.shape)
	print(labels.shape)

	return [imgs, labels]

def get_data(path):
	data = read_json(path)
	imgs = []
	labels = []

	for d in tqdm(data):
		label = get_label(d)
		img = cv2.imread(d)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# img = img.reshape(224,224)
		img = cv2.resize(img, (224, 224))
		labels.append(label)
		imgs.append(img)
	
	imgs = np.array(imgs)
	labels = np.array(labels)

	# imgs[imgs < 70] = 0

	imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1]*imgs.shape[2])
	labels = labels.reshape(labels.shape[0], 1, 1)

	return [imgs, labels]
#include <linux/gpio.h>