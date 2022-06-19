# U2CBERT

**Title** : U2CBERT: Uniformity-aware Knowledge Distillation of BERT with Unsupervised Contrastive learning

For training U2CBERT
```bash
  python main.py --RKD_tp="angle" --bth=3 --ep=100  --p_train="True" --lr=0.00001 --save="saved_file"  --gpu="gpu_num" 
  ```

For fine-turing
```bash
  python main.py --RKD_tp="angle" --bth=32 --ep=300  --f_train="True" --f_task="mrpc" --lr=0.000001 --save="saved_file" --gpu="gpu_num" 
  ```