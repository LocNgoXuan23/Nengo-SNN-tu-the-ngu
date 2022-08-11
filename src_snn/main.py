from models import *
from train_cnn_models import *
from dataset import *
from predict_snn_models import *
from time import time

CKTP_NAME = f'mobilenet_{2}'

if __name__ == '__main__':
	train_data = get_data_cnn('../data/train.json')
	test_data = get_data_cnn('../data/val.json')
	print("LOAD DATA CNN DONE !!")

	model, input, output, conv0 = mobilenet((224,224,1),9)

	trainer(model, train_data, test_data, CKTP_NAME, epochs=300)

	train_data = get_data('../data/train.json')
	test_data = get_data('../data/val.json')
	print("LOAD DATA SNN DONE !!")

	predict_snn(
		model=model, 
		input=input, 
		output=output, 
		conv0=conv0, 
		cktp_name=CKTP_NAME, 
		test_data=test_data, 
		s=0.008, 
		scale=1000, 
		n_steps=300
	)





