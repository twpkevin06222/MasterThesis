# Classification of Clinically Significant Prostate Cancer with Multiparametric MRI
Implementation of master thesis entitled "Classification of Clinically Significant Prostate Cancer with Multiparametric MRI", using deep metric learning method, specifically margin-based softmax loss ([ArcFace Loss](https://arxiv.org/abs/1801.07698)) for discriminative learning to reduce false positive for prostate patients biopsy. Suboptimally, similarity distance learnt by margin-based softmax loss will be extended to content based image retrieval for visual similarity task. 

## Experiment Setup 

### Backbone Model
The backbone model is a 3D classification model, Resnet10, an extension of 3D classification model zoo from [ZFTurbo](https://github.com/ZFTurbo/classification_models_3D). Or simply `pip install classification-models-3D`. 

### ArcFace Loss 
<p align="center">
<img src="fig/ArcFace.png" width="500" height="150">
</p> 

Adapted margin-based softmax loss can be found in `metric_loss.py`. Illustration of the implementation of ArcFace loss is as depcited above ([Image Source](https://github.com/deepinsight/insightface/tree/master/recognition).

### Late Fusion
<p align="center">
<img src="fig/LateFushion.png" width="400" height="300">
</p> 

Three prostate MRI image sequences (T2, DWI, ADC) are provided in the data set.
While it is common practice to liaise the image sequences as channel input for richer
data features, however, it is not possible for all three image sequences to have the
same alignment due to constraints. We hypothesise that each image sequence would
contribute to different representations in the embedding space, by opting to feed
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
* Download dependencies `$ pip install --r requirements.txt`

## Results 
### Classification Task 
|Baseline|ArcFace|
|:--:|:--:|
|![](fig/SingleDenseHead_AUC.png)|![](fig/SingleArcHead_AUC.png)|

|       | AUC    |Accuracy| Precision| Recall | F1|
| :-----------: | :----------: | :-----------: | :-----------: |:-----------: | :-----------: |
| Baseline  | 0.77 (0.01)  | 0.71 (0.02) | 0.72 (0.02) | 0.71 (0.02) | 0.70 (0.03)|
| ArcFace  | 0.79 (0.01)  | 0.73 (0.01) | 0.73 (0.02) | 0.73 (0.01) | 0.72 (0.01)|



### Retrieval Task 
|Baseline|ArcFace|
|:--:|:--:|
|![](fig/SingleDenseHead_tsne.png)|![](fig/SingleArcHead_tsne.png)|

|       | R@1    | R@10_Precision| MAP@10| 
| :-----------: | :----------: | :-----------: | :-----------: |
| Baseline  | 0.57 (0.02)  | 0.58 (0.02) | 0.43 (0.02) |
| ArcFace  | 0.65 (0.02)  | 0.64 (0.02) | 0.51 (0.03) |

### Content Based Image Retrieval (CBIR)
Content-based Image Retrieval (CBIR) attempts to utilise image as a query to retrieve
images from the image database in replacement of keywords for more efficient retrieval
process. In this work, we first map volumetric images to
low dimensional embedding space, where the similarity scores between each query
embedding and retrieval embedding are computed. The nearest neighbour of the
query embedding are ranked according to the similarity scores in descending order.
While it is not trivial to visualise volumetric data and its retrieved images, the mid
slice of every patients’ MRI images are chosen to be displayed for CBIR task with the
assumption that the mid image slice contains the most significant image description
for a patient’s prostate. 

<p align="center">
<img src="fig/SingleArcHead_CBIR.png" width="500" height="500">
</p> 

