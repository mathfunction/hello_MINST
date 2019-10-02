# Hello Deep-Learning , MINST+CNN 範例



## MINST Data載點:  

http://yann.lecun.com/exdb/mnist/  (已下載至 ./data)

## 論文連結:   

<http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>  (古老經典文獻)



## minst_pytorch.py 



#### Python 需要套件:

-  torch  (需要 1.X.X 版才不會有 bug )

-  idx2numpy  (讀取 .idx1-ubtye, .idx3-ubyte )

-  numpy
- matplotlib

#### 命令列 : CPU/GPU訓練 , CPU 推論 , pkl 轉 onnx:

```bash
python minst_pytorch.py --training-CPU [epochs] [batch_size] [lr]
python minst_pytorch.py --training-GPU [epochs] [batch_size] [lr]
python minst_pytorch.py --inference-CPU
python minst_pytorch.py --pkl2onnx [batch_size]
```

其中:  [epochs]  整數(ex:50)，[batch_size] 整數(ex:30)，[lr] 0~1的實數 (ex:0.002)

 





