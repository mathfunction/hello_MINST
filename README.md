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



#### Python : CPU/GPU訓練 + CPU 推論:

```bash
python minst_pytorch.py --training-CPU
python minst_pytorch.py --training-GPU
python minst_pytorch.py --inference
```

#### 





