# Classification of Clinically Significant Prostate Cancer with Multiparametric MRI
Implementation of master thesis entitled "Classification of Clinically Significant Prostate Cancer with Multiparametric MRI", using deep metric learning method, specifically margin-based softmax loss ([ArcFace Loss](https://arxiv.org/abs/1801.07698)) for discriminative learning to reduce false positive for prostate patients biopsy. Suboptimally, similarity distance learnt by margin-based softmax loss will be extended to content based image retrieval for visual similarity task. 

## Experiment Setup 

### Backbone Model
The backbone model is a 3D classification model, Resnet10, an extension of 3D classification model zoo from [ZFTurbo](https://github.com/ZFTurbo/classification_models_3D). Or simply `pip install classification-models-3D`. 

### ArcFace Loss 
<p align="center">
<figure>
	<img src="fig/ArcFace.png" width="500" height="150">
	<figcaption> 
	[Image Source](https://github.com/deepinsight/insightface/tree/master/recognition)
	</figcaption>
</figure>
</p> 

Adapted margin-based softmax loss can be found in `metric_loss.py`.

### Late Fusion
<p align="center">
<img src="fig/LateFushion.png" width="400" height="300">
</p> 

Three prostate MRI image sequences (T2, DWI, ADC) are provided in the data set.
While it is common practice to liaise the image sequences as channel input for richer
data features, however, it is not possible for all three image sequences to have the
same alignment due to constraints. We hypothesise that each image sequence would
contribute to di↵erent representations in the embedding space, by opting to feed
the network on each image sequence respectively. The embedding output of three
image sequences is concatenated forming an ensemble embedding that would enrich
the embedding space. This form a late fusion where the concatenated embedding is
connected to a fully connected network with a dimension size of 128 associating with
a dropout layer before connecting to output layer.

## Repo Structure
* Data granularity of the data set can be found in the folder `/eda`
* The source code for models training can be found in the folder `/train`
* Inferencing of the models can be found in the folder `/inference` 
* The configurations for training and inferencing with cross validation in `.yaml` are listed in `/sweep`

## Requirement
`$ pip install --r requirements.txt`
