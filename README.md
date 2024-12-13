# SinSR Dual Scale

This project is a variasion of [SinSR: Diffusion-Based Image Super-Resolution in a Single Step](https://github.com/wyf0912/SinSR).  
If you want to know more about SinSR, please refer to the original repository.

We modified the loss function of the original SinSR to improve the performance of the model.

## Modifications
The original SinSR uses the following loss function:

$$
\mathcal{L} = \mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{reverse}} + \mathcal{L}_{\text{gt}}
$$

We modified the loss function as follows:

$$
\mathcal{L} = \mathcal{L}_{\text{distill}} + \mathcal{L}_{\text{reverse}} + \mathcal{L}_{\text{gt}} + \mathcal{L}_{\text{crop-gt-distill}} + \mathcal{L}_{\text{crop-gt-reverse}} + \mathcal{L}_{\text{crop-gt}}
$$

where  
$\text{crop-gt}$: is the crop of the ground truth image  
$\mathcal{L}_{\text{distill}}$: is similar to the original distillation loss, changing the input to the crop of the ground truth image  
$\mathcal{L}_{\text{reverse}}$: is similar to the original reverse loss, changing the input to the crop of the ground truth image  
$\mathcal{L}_{\text{crop-gt}}$: is the loss difference between the crop of the ground truth image and the image up-scaled and down-scaled by the model


## Requirements
* Python 3.10, Pytorch 2.1.2, [xformers](https://github.com/facebookresearch/xformers) 0.0.23
* More detail (See [environment.yml](environment.yml))
A suitable [conda](https://conda.io/) environment named `resshift` can be created and activated with:

```
conda env create -n SinSR python=3.10
conda activate SinSR
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
conda activate SinSR
```

## Demo
You can try our method through an online demo:
```sh
python app.py
```

(The time taken for the initial run of the model includes loading the model. Besides, it includes a significant amount of time overhead apart from the algorithms itself, e.g., I/O cost, and web frameworks.)


## Fast Testing
```sh
python3 inference.py -i [image folder/image path] -o [result folder] --ckpt weights/SinSR_v1.pth --scale 4 --one_step
```
