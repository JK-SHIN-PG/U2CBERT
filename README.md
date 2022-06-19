# U2CBERT

**Title** : U2CBERT: Uniformity-aware Knowledge Distillation of BERT with Unsupervised Contrastive learning

### Implementation

- **Environment**
  * os : Ubuntu 18.04  
  * Python >= 3.7  
- **Dependency** 
  * Pytorch >= 1.7 (our developement environment)   
  * pytorch_pretrained_bert
  * numpy
  * pandas
  * scipy
  * torchmetrics


For training U2CBERT
```bash
  python main.py --RKD_tp="angle" --bth=3 --ep=100  --p_train="True" --lr=0.00001 --save="saved_file"  --gpu="gpu_num" 
  ```

For fine-turing
```bash
  python main.py --RKD_tp="angle" --bth=32 --ep=300  --f_train="True" --f_task="mrpc" --lr=0.00001 --save="saved_file" --gpu="gpu_num" 
  ```