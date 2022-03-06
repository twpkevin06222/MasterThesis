# Classification of Clinically Significant Prostate Cancer with Multiparametric MRI
Implementation of master thesis entitled "Classification of Clinically Significant Prostate Cancer with Multiparametric MRI", using deep metric learning method, specifically margin-based softmax loss ([ArcFace Loss](https://arxiv.org/abs/1801.07698)) for discriminative learning to reduce false positive for prostate patients biopsy. Suboptimally, similarity distance learnt by margin-based softmax loss will be extended to content based image retrieval for visual similarity task. 

## Experiment Setup 

### Backbone Model
The backbone model is a 3D classification model, Resnet10, an extension of 3D classification model zoo from [ZFTurbo](https://github.com/ZFTurbo/classification_models_3D). Or simply `pip install classification-models-3D`. 

### ArcFace Loss 
Adapted margin-based softmax loss can be found in `metric_loss.py`.

### Late Fusion
<p align="center">
<img src="fig/LateFushion.png" width="300" height="300">
</p> 
