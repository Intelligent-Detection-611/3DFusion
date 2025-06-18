# 3DFuion Version 0.1

The initial version of 3DFusion modifies the traditional 2D multimodal image fusion network to enable training on 3D medical data in NIfTI format.


## Recommended Environment(win10 1080Ti)

To be continued...

More detailed environment requirements can be found in ```requirements.txt```. 

## To Training

### 1. Convert MambaMorph_Data file to image form

Process the MambaMorph_Data and convert it into registered image fusion data.

For easy viewing of images and debugging code, plz run the following code:
```shell
python trans_illum_data.py --h5_path 'datasets/data_illum.h5' --cls_root_path 'datasets/cls_dataset'
```
The converted directory format is as follows:
```shell
 cls_dataset/
 ├── day
 │   ├── day_0.png
 │   ├── day_1.png
 │   ├── ......
 ├── night
 │   ├── night_0.png
 │   ├── night_1.png
 │   ├── ......
```


 ### 2. Convert data_MSRS.h5 file to image form
The dataset for training the illumination-aware fusion network can be download from [data_MSRS.h5](https://pan.baidu.com/s/1cO_wn2DOpiKLjHPaM1xZYQ?pwd=PIAF).

For easy viewing of images and debugging code, plz download the file and run the following code:
```shell
python trans_msrs_data.py --h5_path 'datasets/data_MSRS.h5' --msrs_root_path 'datasets/msrs_train'
```

The converted directory format is as follows:
```shell
 msrs_train/
 ├── Inf
 │   ├── 0.png
 │   ├── 1.png
 │   ├── ......
 ├── Vis
 │   ├── 0.png
 │   ├── 1.png
 │   ├── ......
```

If the link given above has expired, you can download the dataset [here](https://pan.baidu.com/s/18XjhLlzr_t9Y1sDYudJHww?pwd=u1tt). 


### 3. Training the Illumination-Aware Sub-Network
```shell
python train_illum_cls.py --dataset_path 'datasets/cls_dataset' --epochs 100 --save_path 'pretrained' --batch_size 128 --lr 0.001
```
Then the weights of the best classification model can be found in [pretrained/best_cls.pth](https://github.com/linklist2/PIAFusion_pytorch/blob/master/pretrained/best_cls.pth), The test accuracy of the best model is around 98%.

### 4. Training the Illmination-Aware Fusion Network
```shell
python train_fusion_model.py --dataset_path 'datasets/msrs_train' --epochs 30 --cls_pretrained 'pretrained/best_cls.pth' --batch_size 128 --lr 0.001 --loss_weight '[3, 7, 50]' --save_path 'pretrained'
```
The values in **loss_weight** correspond to **loss_illum**, **loss_aux**, **gradinet_loss** respectively.


## To Testing
### 1. The MSRS Dataset
```shell
python test_fusion_model.py --h5_path 'test_data/MSRS' --save_path 'results/fusion' --fusion_pretrained 'pretrained/fusion_model_epoch_29.pth'
```

The fusion result can be found in the directory corresponding to the ```save_path``` parameter.

It can be observed that the results is not particularly ideal and needs to be further adjusted.

**Note: The directory structure of the test dataset should be the same as that of the training dataset, as follows**:



```shell
 MSRS/
 ├── Inf
 │   ├── 00537D.png
 │   ├── ......
 ├── Vis
 │   ├── 00537D.png
 │   ├── ......
```

# TODO

 - [ ] Test The RoadScene Dataset
 - [ ] Test The TNO Dataset  
 - [ ] Adjust the loss factor parameter
 - [ ] Modify the loss function


## If this work is helpful to you, please cite it as：
```

```
