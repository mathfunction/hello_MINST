# Hello Deep-Learning  MINST 範例



## MINST Data載點:  

http://yann.lecun.com/exdb/mnist/

## 論文連結:   

<http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>



## minst_pytorch_XXX.py 



#### Python 需要套件:

-  torch  (需要 1.X.X 版才不會有 bug )

-  idx2numpy  (讀取 idx3-ubyte )

-  numpy
- matplotlib



#### CPU+Python 訓練:

```bash
python minst_pytorch_cpu.py --training
```

#### CPU+Python 推論:

```bash
python minst_pytorch_cpu.py --inference
```



