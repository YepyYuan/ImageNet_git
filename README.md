## Bird Classification on CUB_200_2011

This repo includes a mid-term project of _DATA620004, School of Data Sciences, Fudan University_.

### Project Introduction
This bird classification is a fine-tuned model based on ResNet-18. We loaded an ImageNet pretrained resnet18 model (``torchvision.models.resnet18``), and then modified the full-connected layer to make it refer to our task. We trained the fine-tuned model with CUB_200_2011 bird dataset (downloaded from: https://data.caltech.edu/records/65de6-vp158). This dataset includes 200 categories and 11,788 images.Training dataset and testing dataset has 5994 images and 5794 images, respectively.

### Codes
file tree
```
│  dataloader.py
│  inference.py
│  main.ipynb
│  README.md
│  res18.py
│  split.py
│  test_images.txt
│  train.py
│  train_images.txt
│
├─data
│
└─models
```

`dataloader.py`: Construct a `Dataset` class to transfer image data to tensor and service for `DatasetLoader` to load later.

`res18.py`: Load `resnet18()` from `torchvision.models`. It can load either pretraied model or randomlt initialized model.

`split.py`: Split the dataset into training part and testing part, according to `train_test_split.txt`

`train.py`: Implement the code for training

`inference.py`: Code for inference

`main.py`: Implemention for the whole progress