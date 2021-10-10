# Content:
## 1. [Description](#Description)
## 2. [Start ADAS](#Start)
## 3. [Segmentation](#Segmentation)
## 4. [Object detection](#Object-detection)

### Description:
### This repository resolve multiple task:
1. Segmentation of the current and other road lanes;
2. Detection of objects in front of the machine based on the kitty dataset. 

### Start
### For start set config in config files.
### Download weights and datasets:
```python
python download_dataset_and_weights.py
``` 
### And run main script:
```python
python main.py
```
### Segmentation:
### Weights for net:
1. [U2Net]()
2. [UNet]() with `Efficientnet-b0` as backbone from library [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)

### Object detection
### *(In progress)*