"""
	開發中檔案 : openvino 有 bug 請不要使用 !! 
	
	This is Tensorflow 2.0 keras MINST run on CPU/GPU code

	Download MINST from :  http://yann.lecun.com/exdb/mnist/
	
	套件列表:
		tensorflow 2.0.0 深度學習框架 
		idx2Numpy (格式轉檔)
		matplotlib (畫樣本)
	
	#================================================================================================================#
	openvino 程式碼更動:
		1.deployment_tools/model_optimizer/mo/front/tf/partial_infer/tf.py , 147 行 tf.NodeDef 改成 tf.compat.v1.NodeDef
		2.deployment_tools/model_optimizer/mo/front/tf/loader.py , 
		
			tf.NodeDef 改成 tf.compat.v1.NodeDef 
			tf.GraphDef 改成 tf.compat.v1.GraphDef
			tf.MetaGraphDef 改成 tf.compat.v1.MetaGraphDef
		3. deployment_tools/model_optimizer/mo/utils/tensorboard.py	, 26 行

	#================================================================================================================#


"""
import sys
import os 
import numpy as np
import subprocess
import shutil
import matplotlib.pyplot as plt
# tensorflow
import tensorflow as tf
from tensorflow import keras




# 取得該檔案絕對路徑
ABSPATH = os.path.dirname(os.path.abspath(__file__))
#=============================================================================================================================================
# 讀檔 class ，特殊檔案 : API 轉成 numpy 轉 torch.tensor : 決定 x,y 的長相
#=============================================================================================================================================
class ReadMINST:
	def __init__(self):
		import idx2numpy
		self.xtrain = tf.constant(idx2numpy.convert_from_file(ABSPATH+"/data/train-images.idx3-ubyte"),dtype=tf.float32)
		self.ytrain = tf.constant(idx2numpy.convert_from_file(ABSPATH+"/data/train-labels.idx1-ubyte"),dtype=tf.int64)
		self.xtest = tf.constant(idx2numpy.convert_from_file(ABSPATH+"/data/t10k-images.idx3-ubyte"),dtype=tf.float32)
		self.ytest = tf.constant(idx2numpy.convert_from_file(ABSPATH+"/data/t10k-labels.idx1-ubyte"),dtype=tf.int64)
		print("-------- 資料形狀 -----------")
		print("x_train:{}".format(self.xtrain.get_shape()))
		print("y_train:{}".format(self.ytrain.get_shape()))
		print("x_test:{}".format(self.xtest.get_shape()))
		print("y_test:{}".format(self.ytest.get_shape()))
		print("----------------------------")



	# 把資料印出來看 
	def look(self,idx=0,_boolTrain=True):
		if _boolTrain == True:
			print("這是_訓練資料_第 {} 個樣本 , Label = {} [關閉視窗後會繼續]".format(idx+1,self.ytrain[idx]))
			mat = self.xtrain[idx]
		else:
			print("這是_測試資料_第 {} 個樣本 , Label = {} [關閉視窗後會繼續]".format(idx+1,self.ytest[idx]))
			mat = self.xtest[idx]
		plt.matshow(mat,cmap=plt.get_cmap('gray'))
		plt.show()
		

#============================================================================================================================================
# tensorflow 模型 : 
#============================================================================================================================================
def SimpleNN():
	return keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128,activation="relu"),
		keras.layers.Dense(10,activation="softmax")
	])


class TrainingEngine:
	def __init__(self,_boolGPU=False):
		if _boolGPU == True:
			pass
		else:
			# onlyCPU
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		self.model = SimpleNN()
		

	def run(self,epochs,batch_size,lr,_boolContinue):
		if _boolContinue == 1:
			shutil.copyfile(ABSPATH+"/model/SimpleNN.pb",ABSPATH+"/model/saved_model.pb")
			self.model = keras.models.load_model(ABSPATH+"/model/")
			os.remove(ABSPATH+"/model/saved_model.pb")
			print("___________ Load Tensorflow Model From .pb _____________")

		data = ReadMINST()
		data.look(0)
		adam = keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=adam,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
		self.model.fit(data.xtrain,data.ytrain,epochs=epochs,batch_size=batch_size,validation_data=(data.xtest,data.ytest))
		#---------------------------------------------------------------------------------
		# save model : since default tensorflow can't not change filename "saved_model.pb"
		print("Save Tensorflow Model As .pb !!")
		keras.models.save_model(self.model,ABSPATH+"/model")
		shutil.copyfile(ABSPATH+"/model/saved_model.pb",ABSPATH+"/model/SimpleNN.pb")
		os.remove(ABSPATH+"/model/saved_model.pb")
		#-----------------------------------------------------------------------------------


class InferenceEngine:
	def __init__(self,_boolGPU=False):
		if _boolGPU == True:
			pass
		else:
			# onlyCPU
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		data = ReadMINST()
		#----------------------------------------------------------------------------------
		# load model : since default tensorflow can't not change filename "saved_model.pb"
		shutil.copyfile(ABSPATH+"/model/SimpleNN.pb",ABSPATH+"/model/saved_model.pb")
		self.model = keras.models.load_model(ABSPATH+"/model/")
		os.remove(ABSPATH+"/model/saved_model.pb")
		print("___________ Load Tensorflow Model From .pb _____________")
		#----------------------------------------------------------------------------------
		self.model.summary()
		print("Check MINST-TrainData-Accuracy: {} %".format(self.score_func(tf.argmax(self.model(data.xtrain),axis=1),data.ytrain))) 
		print("Check MINST-TestData-Accuracy: {} %".format(self.score_func(tf.argmax(self.model(data.xtest),axis=1),data.ytest)))
	#-----------------------------------------------------
	# just scan the tf.tensor compute accuracy
	def score_func(self,y_pred,y_target):
		_count = 0 
		for i in range(len(y_target)):
			if y_target[i] == y_pred[i]:
				_count += 1
		return _count*100/len(y_target)









def ModelOptimizerOpenVINO():
	if os.name == "nt":
		os.chdir("C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\model_optimizer")
	else:
		os.chdir("/opt/intel/openvino/deployment_tools/model_optimizer")
	#=============================================================================================================================================
	# script 指令
	#============================================================================================================================================
	try:
		subprocess.run(["python3","mo.py","--input_model",ABSPATH+"/model/SimpleNN.pb","--input_model_is_text","--output_dir",ABSPATH+"/model/","--data_type","FP32"])	
	except FileNotFoundError:
		subprocess.run(["python","mo.py","--input_model",ABSPATH+"/model/SimpleNN.pb","--input_model_is_text","--output_dir",ABSPATH+"/model/","--data_type","FP32"])	
	os.chdir(ABSPATH)


def command():
	print("==================================================================")
	print("[Tensorflow]")
	print("\t python minst_tensorflow.py --training-CPU [epochs] [batch_size] [lr] [continue?]")
	# need CUDA 10 + cuDNN >= 7.4.1
	#print("\t python minst_tensorflow.py --training-GPU [epochs] [batch_size] [lr] [continue?]")
	print("\t python minst_tensorflow.py --inference-CPU")
	print("==================================================================")
	#print("[OpenVINO] pb  ---> xml,bin ---> Intel IE")
	#print("\t python minst_pytorch.py --model-optimizer ")
	print("==================================================================")

if __name__ == "__main__":
	if len(sys.argv) == 6:
		epochs = int(sys.argv[2])
		batch_size = int(sys.argv[3])
		lr = float(sys.argv[4])
		if sys.argv[1] == "--training-CPU":
			TrainingEngine(False).run(epochs,batch_size,lr,int(sys.argv[5]))
		elif sys.argv[1] == "--training-GPU":
			# GPU only
			TrainingEngine(True).run(epochs,batch_size,lr,int(sys.argv[5]))
		else:
			command()
	elif len(sys.argv) == 2:
		if sys.argv[1] == "--inference-CPU":
			InferenceEngine(False)
		elif sys.argv[1] == "--model-optimizer":
			ModelOptimizerOpenVINO()
		else:
			command()
	else:
		command()

	
	