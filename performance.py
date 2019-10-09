"""
	performance comparsion




"""
import timeit
from minst_pytorch import *





def pretraining():	
	TrainingEngine(False).run(nepochs=30,batch_size=100,lr=0.005,_boolContinue=0) # 訓練端
	InferenceEngine().to_onnx(batch_size=1)
	



def performance(_boolOpenVINO=False):
	
	data = ReadMINST(False)
	n_train = data.xtrain.size(0)
	n_test = data.xtest.size(0)
	#=========== start to run ====================#
	if _boolOpenVINO == False:
		on_pure_pytorch = InferenceEngine()
		t1 = timeit.default_timer()
		for i in range(n_train):
			x = data.xtrain[i]
			on_pure_pytorch.infer(x.view(1,x.size(0),x.size(1)))
		for i in range(n_test):
			x = data.xtest[i]
			on_pure_pytorch.infer(x.view(1,x.size(0),x.size(1)))  
		t2 = timeit.default_timer()

		print("[{}],time_cost:{}s".format("pure_pytorch",t2-t1))
	else:
		on_openvino = InferenceEngineOpenVINO(1)
		t1 = timeit.default_timer()
		for i in range(n_train):
			x = data.xtrain[i]
			on_openvino.infer(x.numpy().reshape(1,28,28))
		for i in range(n_test):
			x = data.xtest[i]
			on_openvino.infer(x.numpy().reshape(1,28,28))
		t2 = timeit.default_timer()
		print("[{}],time_cost:{}s".format("openvino_pytorch",t2-t1))






if __name__ == "__main__":
	#pretraining()
	performance(_boolOpenVINO=False)
	ModelOptimizerOpenVINO("FP32")
	performance(_boolOpenVINO=True)
	ModelOptimizerOpenVINO("FP16")
	performance(_boolOpenVINO=True)
	