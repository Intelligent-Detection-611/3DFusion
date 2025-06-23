# 3DFusion Version 0.1

The initial version of 3DFusion modifies the traditional 2D multimodal image fusion network to enable training on 3D medical data in NIfTI format.


## Recommended Environment

To be continued...

More detailed environment requirements can be found in ```requirements.txt```. 

## To Training

### 1. Convert MambaMorph_Data file to image form

Process the MambaMorph_Data and convert it into registered image fusion data.

The original data in the **volumes_center** folder should be processed by executing the following preprocessing scripts in sequence to generate the data format required for model training.

#### Dataset Splitting

Run the [split_dataset.py](https://github.com/Intelligent-Detection-611/3DFusion/blob/main/utils/split_dataset.py) script to divide the original dataset into training sets.(test and val sets are not prepared)

```shell
python split_dataset.py 
```

#### File Renaming

Run the [rename.py](https://github.com/Intelligent-Detection-611/3DFusion/blob/main/utils/rename.py) script to rename the divided files to comply with the data loader’s reading specifications.

```shell
python rename.py
```

The converted directory format is as follows:
```shell
 nifti_train/
 ├── ct
 │   ├── 1BA001.nii.gz
 │   ├── 1BA005.nii.gz
 │   ├── ......
 ├── mr
 │   ├── 1BA001.nii.gz
 │   ├── 1BA005.nii.gz
 │   ├── ......
```

### 2. Training the 3D Fusion Network
#### (Specific information is to be added.)
```shell
python train_3D.py
```

## To Testing
### (Specific information is to be added.)
```shell
python test_fusion_model_3D.py 
```

# TODO


## If this work is helpful to you, please cite it as：
```

```
test
