# Hello DL / MINST / CNN / Pytorch / OpenVINO / Python Code 

#### Python 需要套件:

- torch  (需要 1.X.X 版才不會有 bug )
- idx2numpy  (讀取 .idx1-ubtye, .idx3-ubyte )
- numpy 
- matplotlib
- onnx (深度學習模型通用格式)  https://onnx.ai/
- Intel - openVINO  https://software.intel.com/en-us/openvino-toolkit

### 使用方法:

```bash
git clone https://github.com/mathfunction/hello_MINST.git
python minst_pytorch.py 
```



## MINST Data載點:  

http://yann.lecun.com/exdb/mnist/  (已經存在至 ./data)

## MINST 論文連結:   

<http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>  (古老經典文獻)



## minst_pytorch.py:

### Python 程式區塊

```python
class ReadMINST:    			# 關於讀資料(訓練/測試) MINST 相關
class SimpleCNN:    			# 關於使用 pytorch 建構 深度學習 CNN 模型
class TrainingEngine:  			# 關於使用 pytorch 訓練模型 , 訓練過程
class InferenceEngine:			# 關於使用 pytorch 匯入模型 , 推論過程 , 匯出 onnx 檔
class InferenceEngineOpenVINO:  # 關於使用 openVINO InferenceEngine python API 程式碼
def ModelOptimizerOpenVINO():   # 關於使用 openVINO ModelOptimizer 從.onnx -->.xml.bin script 
if __name__ == "__main__":      # 關於命令列功能
    
```



#### 命令列 : Pytorch CPU/GPU訓練 , CPU 推論 :

```bash
python minst_pytorch.py --training-CPU [epochs] [batch_size] [lr] [continue?]
python minst_pytorch.py --training-GPU [epochs] [batch_size] [lr] [continue?]
python minst_pytorch.py --inference-CPU
```

其中:  

​	[epochs]  整數(ex:50)

​	[batch_size] 整數(ex:30)

​	[lr] 0~1的實數 (ex:0.002) 

​	[continue?]  0:重新訓練   1: 匯入 SimpleCNN.pkl 繼續訓練

--------------

#### OpenVINO 安裝教學連結

- Windows:

  https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html>

- Linux/Mac:

  <https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html>

#### OpenVINO 激活環境指令 (使用OpenVINO前，需要下這指令) 

##### - Windows:

```bash
"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
```

##### - Linux/Mac:

```bash
source /opt/intel/openvino/bin/setupvars.sh
```



#### torch.onnx 可把 SimpleCNN.pkl 轉成 SimpleCNN_Batch1.onnx

```bash
python minst_pytorch.py --pkl2onnx 1 
```

#### openVINO-ModelOptimizer 可把 SimpleCNN_Batch1.onnx 轉成 .xml , .bin

```bash
python minst_pytorch.py --model-optimizer 
```



#### openVINO-InferenceEngine Python API , CPU 推論 (BatchSize=1)

```bash
python minst_pytorch.py --inferenceOpenVINO_Batch1-CPU
```







