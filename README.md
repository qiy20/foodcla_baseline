# 1.项目目录

```
foodcla
│  baseline.py
│  list.txt
│  README.md
│  test_qtcom.txt
│  train_qtcom.txt
│  utils.py
│  val_qtcom.txt
│        
├─run        
├─test_new
├─Train_qtc
└─val

```

# 2.运行

运行参数见run/exp/opt.yaml

```bash
python baseline.py --adam
```

# 3.结果

### 3.1 resnet50(64 epochs)

top1_acc:0.4463

top5_acc:0.7180

具体结果见run/exp

### 3.2 resnet101(128 epochs )

top1_acc:0.5372

top5_acc:0.7806

具体结果见run/exp2

### 3.3 resnet101(fine tune)

top1_acc:0.6649

top5_acc:0.8599

具体结果见run/exp3
