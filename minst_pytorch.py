"""

	This is Pytorch MINST run on CPU/GPU code

	Download MINST from :  http://yann.lecun.com/exdb/mnist/
	
	套件列表:
		torch   (pytorch 深度學習框架)
		idx2Numpy (格式轉檔)
		matplotlib (畫樣本)
		

"""
import sys
import os 
import numpy as np
import subprocess
import matplotlib.pyplot as plt
# pytorch
import torch
import torch.nn as nn
import torch.utils.data as Data



# 取得該檔案絕對路徑
ABSPATH = os.path.dirname(os.path.abspath(__file__))
#=============================================================================================================================================
# 讀檔 class ，特殊檔案 : API 轉成 numpy 轉 torch.tensor : 決定 x,y 的長相
#=============================================================================================================================================
class ReadMINST:
	def __init__(self,_boolPrint=True):
		import idx2numpy
		# check carefully : from_numpy()  will return torch.tensor( requires_grad=False ) , data is not weights 
		self.xtrain = torch.from_numpy(idx2numpy.convert_from_file(ABSPATH+"/data/train-images.idx3-ubyte")).type(torch.FloatTensor)
		self.ytrain = torch.from_numpy(idx2numpy.convert_from_file(ABSPATH+"/data/train-labels.idx1-ubyte")).type(torch.LongTensor)
		self.xtest = torch.from_numpy(idx2numpy.convert_from_file(ABSPATH+"/data/t10k-images.idx3-ubyte")).type(torch.FloatTensor)
		self.ytest = torch.from_numpy(idx2numpy.convert_from_file(ABSPATH+"/data/t10k-labels.idx1-ubyte")).type(torch.LongTensor)
		if _boolPrint == True:
			print("-------- 資料形狀 -----------")
			print("x_train:{}".format(self.xtrain.shape))
			print("y_train:{}".format(self.ytrain.shape))
			print("x_test:{}".format(self.xtest.shape))
			print("y_test:{}".format(self.ytest.shape))
			print("----------------------------")
	# 把資料印出來看 
	def look(self,idx=0,_boolTrain=True):
		if _boolTrain == True:
			print("這是_訓練資料_第 {} 個樣本 , Label = {} [關閉視窗後會繼續]".format(idx+1,self.ytrain[idx].item()))
			mat = self.xtrain[idx]
		else:
			print("這是_測試資料_第 {} 個樣本 , Label = {} [關閉視窗後會繼續]".format(idx+1,self.ytest[idx].item()))
			mat = self.xtest[idx]
		plt.matshow(mat,cmap=plt.get_cmap('gray'))
		plt.show()
		

#============================================================================================================================================
# pytorch 模型 : 決定 f 的長相
#============================================================================================================================================
# CNN 詳細可參考 reference  "https://pytorch.org/docs/stable/nn.html#convolution-layers"
class SimpleCNN(nn.Module):
	# 定義 架構種類 !!
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
			nn.Conv2d(
				in_channels=1,              # input height
				out_channels=16,            # n_filters
				kernel_size=5,              # filter size
				stride=1,                   # filter movement/step
				padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
			),                              # output shape (16, 28, 28)
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
		)
		self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=5,
				stride=1,
				padding=2
			),     # output shape (32, 14, 14)
			nn.ReLU(),                      # activation
			nn.MaxPool2d(2),                # output shape (32, 7, 7)
		)
		self.out = nn.Linear(32*7*7, 10)   # fully connected layer, output 10 classes
	# 定義 架構運算先後 !!
	def forward(self, x):
		x = x.view(x.size(0),-1,x.size(1),x.size(1))# 因為 conv API 需要 NCHW tensor 
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		y_pred = self.out(x)                # output shape (batch_size,10)
		return y_pred 
#==================================================================================================================================
# 決定 loss function , score function , optimizer , how to train , when to stop !! 
#==================================================================================================================================

class TrainingEngine:
	def __init__(self,_boolGPU=False):
		self._boolGPU = _boolGPU
		self.model = SimpleCNN()
		# 選擇 loss_func
		self.loss_func = torch.nn.CrossEntropyLoss()  
	# y_target in {0,1,2,3,4,....9}
	def score_func(self,y_pred,y_target):
		# 利用 torch.argmax((batch x 10),dim=1) ---> ( batch  ) 把 one-hot 轉成 位址整數 
		y_pred_int = torch.argmax(y_pred,dim=1)
		# accuracy
		_count = 0 
		for i in range(y_pred_int.size(0)):
			if y_target[i] == y_pred_int[i]:
				_count += 1
		return _count
	#==============================================================================================================#

	#==============================================================================================================#
	def run(self,nepochs=10,batch_size=30,lr=0.005,_boolContinue=0,modelpath=ABSPATH+"/model/"+"SimpleCNN.pkl"):
		if _boolContinue == 1:
			self.model.load_state_dict(torch.load(modelpath))
			print("Load Model from {}".format(modelpath))
		data = ReadMINST()
		n_train = data.xtrain.size(0)
		n_valid = data.xtest.size(0)
		xtest = data.xtest
		ytest = data.ytest
		#############################################
		# GPU
		if self._boolGPU == True: 
			self.model.cuda()
			xtest = xtest.cuda()
			ytest = ytest.cuda()
		#############################################
		data.look(torch.randint(0,n_train,(1,)).item())
		#===========================================================================#
		# 把訓練資料切割成一個個 batchsize : (n,f) ---> (n/b,b, f) 
		#===========================================================================#
		torch_dataset = Data.TensorDataset(data.xtrain,data.ytrain)
		loader = Data.DataLoader(
				dataset=torch_dataset,
				batch_size=batch_size,
				shuffle=True,
				num_workers=2,
			)
		
		# 選擇優化演算法
		self.optimizer = torch.optim.Adam(self.model.parameters(),lr)
		incumbent = 0.5
		print("=====================================================================")
		print(" Start Training & Validation Epochs:{} , BatchSize:{} , LearningRate:{}".format(nepochs,batch_size,lr))
		#===========================================================================#
		# 訓練 + 驗證過程 , 計算 train_loss , valid_loss , train_acc , vaild_acc
		#===========================================================================#
		for epoch in range(nepochs):
			self.model.train()
			train_loss = 0.0
			train_count = 0.0
			# batch-size training 
			for step,(x,y_target) in enumerate(loader):
				###################################
				# GPU
				if self._boolGPU == True:
					x = x.cuda()
					y_target = y_target.cuda()
				##################################
				self.optimizer.zero_grad()
				y_pred = self.model(x)
				loss = self.loss_func(y_pred,y_target)
				train_loss += loss.item()
				train_count += self.score_func(y_pred,y_target)
				loss.backward()
				self.optimizer.step()
				
			# validation 
			self.model.eval()  # 凍結權重
			train_loss/= n_train
			train_acc = train_count/n_train
			valid_loss = self.loss_func(self.model(xtest),ytest).item()
			valid_count = self.score_func(self.model(xtest),ytest)
			valid_loss/= n_valid
			valid_acc = valid_count/n_valid
			print("epoch:{} | train_loss:{} | valid_loss:{} | train_acc:{} | valid_acc:{}".format(epoch+1,train_loss,valid_loss,train_acc,valid_acc))
			#====================================================================================
			# 準確率歷史新高則存"權重" save model
			if valid_acc > incumbent:
				incumbent = valid_acc
				torch.save(self.model.state_dict(),modelpath)
				print("Save {} at valid_acc: {}".format(modelpath,incumbent))
#=======================================================================================================================



class InferenceEngine:
	def __init__(self,modelpath=ABSPATH+"/model/"+"SimpleCNN.pkl"):
		self.modelpath = modelpath
		self.model = SimpleCNN()
		print("===========================================")
		print("Load {}".format(modelpath))
		# 讀取權重檔
		self.model.load_state_dict(torch.load(modelpath))
	# x be  (n,28x28) FloatTensor
	def infer(self,x):
		return self.model(x)

	# 個別從訓練資料 , 測試資料 挑一張預測
	def run(self):
		data = ReadMINST()
		n_train = data.xtrain.size(0)
		n_valid = data.xtest.size(0)

		while 1:
			r1 = torch.randint(0,n_train,(1,)).item()
			# 從 train 挑
			x = data.xtrain[r1]
			y = data.ytrain[r1]
			print("預測結果:{} , 正確結果:{} \t".format(torch.argmax(self.infer(x.view(1,x.size(0),x.size(1))),dim=1).item(),y),end="")
			data.look(r1)
			# 從 test 挑
			r2 = torch.randint(0,n_valid,(1,)).item()
			x = data.xtest[r2]
			y = data.ytest[r2]
			print("預測結果:{} , 正確結果:{} \t".format(torch.argmax(self.infer(x.view(1,x.size(0),x.size(1))),dim=1).item(),y),end="")
			data.look(r2,False)
			_str = input("任何鍵重測 , CTRL+C 可終止程式 !!")
	# 教學 https://michhar.github.io/convert-pytorch-onnx/
	def to_onnx(self,batch_size=1):
		filename = "{}_Batch{}.onnx".format(self.modelpath.split(".")[0],batch_size)
		# 只是確認 model input 形狀 batch_size x 28 x 28 
		dummy_input = torch.randn(batch_size,28,28) 
		torch.onnx.export(self.model,dummy_input,filename)
		print("create {} !!".format(filename))
			



"""============================================================================
	需要安裝　intel-openVINO : 
	Python API 教學:
		https://www.youtube.com/watch?v=6Dzvamu3mg8&list=PLDKCjIU5YH6jMzcTV5_cxX9aPHsborbXQ&index=17 
	激活 :
		"C:/Program Files (x86)/IntelSWTools/openvino/bin/setupvars.bat"

================================================================================"""
class InferenceEngineOpenVINO:
	def __init__(self,batch_size=1):
		from openvino.inference_engine import IENetwork, IEPlugin 
		xmlfile = ABSPATH + "/model/SimpleCNN_Batch{}.xml".format(batch_size)
		binfile = ABSPATH + "/model/SimpleCNN_Batch{}.bin".format(batch_size)
		self.net = IENetwork(model=xmlfile, weights=binfile)
		
		self.plugin = IEPlugin(device="CPU")
		#self.plugin = IEPlugin(device="GPU")  if you have intel GPU not nvidia  
		self.exec_net = self.plugin.load(network=self.net)
		self.input_blob = next(iter(self.net.inputs))
		self.output_blob = next(iter(self.net.outputs))

		print("========================================================")
		print("Load Network Topology : {}".format(xmlfile))
		print("Load Network Weight : {}".format(binfile))
		print("Input_Shape: {}".format(self.net.inputs[self.input_blob].shape))
		print("Output_Shape: {}".format(self.net.outputs[self.output_blob].shape))
		print("========================================================")

	#========================================================================#
	"""
	# note : 
			x = (1,28,28) float numpy array
			self.exec_net.infer(inputs={self.input_blob:x})[self.output_blob] => (1,10) numpy array 
	 		np.argmax(,axis=1) => (1,1) array
	"""
	def infer(self,x):
		return np.argmax(self.exec_net.infer(inputs={self.input_blob:x})[self.output_blob],axis=1)[0]
		
	def run(self):
		data = ReadMINST()
		n_train = data.xtrain.size(0)
		n_valid = data.xtest.size(0)

		while 1:
			#=============================================================
			r1 = torch.randint(0,n_train,(1,)).item()
			# 從 train 挑
			x = data.xtrain[r1].numpy()
			x = x.reshape(1,28,28)
			y = data.ytrain[r1].item()
			print("預測結果:{} , 正確結果:{} \t".format(self.infer(x),y),end="")
			
			
			data.look(r1)
			#============================================================
			# 從 test 挑
			r2 = torch.randint(0,n_valid,(1,)).item()
			x = data.xtest[r2].numpy()
			x = x.reshape(1,28,28)
			y = data.ytest[r2].item()
			print("預測結果:{} , 正確結果:{} \t".format(self.infer(x),y),end="")
			
			
			data.look(r2,False)
			_str = input("任何鍵重測 , CTRL+C 可終止程式 !!")




def ModelOptimizerOpenVINO():
	if os.name == "nt":
		os.chdir("C:\\Program Files (x86)\\IntelSWTools\\openvino\\deployment_tools\\model_optimizer")
	else:
		os.chdir("/opt/intel/openvino/deployment_tools/model_optimizer")
	#=============================================================================================================================================
	# script 指令
	#============================================================================================================================================
	try:
		subprocess.run(["python3","mo.py","--input_model",ABSPATH+"/model/SimpleCNN_Batch1.onnx","--output_dir",ABSPATH+"/model/","--data_type","FP32"])	
	except FileNotFoundError:
		subprocess.run(["python","mo.py","--input_model",ABSPATH+"/model/SimpleCNN_Batch1.onnx","--output_dir",ABSPATH+"/model/","--data_type","FP32"])	
	os.chdir(ABSPATH)





def command():
	print("==================================================================")
	print("[Pytorch]")
	print("\t python minst_pytorch.py --training-CPU [epochs] [batch_size] [lr] [continue?]")
	print("\t python minst_pytorch.py --training-GPU [epochs] [batch_size] [lr] [continue?]")
	print("\t python minst_pytorch.py --inference-CPU")
	print("\t python minst_pytorch.py --pkl2onnx [batch_size]")
	print("==================================================================")
	print("[OpenVINO] pkl ---> onnx ---> xml,bin ---> Intel IE")
	print("\t python minst_pytorch.py --model-optimizer ")
	print("\t python minst_pytorch.py --inferenceOpenVINO_Batch1-CPU")
	print("==================================================================")


if __name__ == "__main__":
	if len(sys.argv) == 6:
		epochs = int(sys.argv[2])
		batch_size = int(sys.argv[3])
		lr = float(sys.argv[4])
		if sys.argv[1] == "--training-CPU":
			TrainingEngine(False).run(epochs,batch_size,lr,int(sys.argv[5])) # 訓練端
		elif sys.argv[1] == "--training-GPU":
			TrainingEngine(True).run(epochs,batch_size,lr,int(sys.argv[5])) # 訓練端
		else:
			command()
	elif len(sys.argv) == 2:
		if sys.argv[1] == "--inference-CPU":
			InferenceEngine().run() # 推論端
		elif sys.argv[1] == "--inferenceOpenVINO_Batch1-CPU":
			InferenceEngineOpenVINO(1).run()
		elif sys.argv[1] == "--model-optimizer":
			ModelOptimizerOpenVINO()
		else:
			command()
	elif len(sys.argv) == 3: 
		if sys.argv[1] == "--pkl2onnx":
			InferenceEngine().to_onnx(int(sys.argv[2]))
		else:
			command()
	else:
		command()

	
	