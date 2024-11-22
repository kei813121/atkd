# Adaptive Temperature Based on Logits Correlation in Knowledge Distillation

## Installation

The repo is tested with Python 3.10.11, PyTorch 2.0.1, and CUDA 11.7.
(Dcoker Image : pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel)

## Running
1. Train a teacher model by : 
    ```
    python3 train_teacher.py --model vgg13
    ```

2. An example of training a student model with our method is given by:
    ```
	python3 train_student.py --path_t ./save/models/cifar100/resnet32x4/resnet32x4_best.pth --model_s resnet8x4 --distill kd --logit_stand --mlogit_temp --kd_T 0 -r 0.1 -b 9 -a 0 --trial 1
    ```
    where the flags are explained as:
    - `--logit_stand`: a flag to apply for z-score standardization
    - `--mlogit_temp`: a flag to use the maximum logit as a temperature

## Acknowledgements

This repo is developed based on the following code:
[RepDistiller](https://github.com/HobbitLong/RepDistiller)
[ITRD](https://github.com/roymiles/ITRD/tree/master)
[logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD)
[CTKD](https://github.com/zhengli97/CTKD)
[DKD](https://github.com/megvii-research/mdistiller)

