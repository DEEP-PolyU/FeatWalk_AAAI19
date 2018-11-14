# Large-Scale Heterogeneous Feature Embedding
Large-Scale Heterogeneous Feature Embedding, AAAI 2019

## Installation
- Requirements
1. numpy
2. scipy
3. gensim
4. sklearn
- Usage
1. cd AANE_Python
2. pip install -r requirements.txt
3. python3.6 Runme.py

## Input and Output
- Input: dataset such as "ACM.mat"
- Output: Embedding.mat, with "H_FeatWalk" denotes the joint heterogeneous feature embedding, and "H_FeatWalk_X" denotes the single feature embedding

## Code in Python
```
from FeatWalk import featurewalk
H = featurewalk(featur1=Feature1, alpha1=.4, featur2=Feature2, alpha2=0.4, Net=Network, beta=0, num_paths=50, path_length=25, dim=100, win_size=5).function()
```

- featur1     is the first feature matrix
- alpha1      is the weight for the first feature matrix, 0 <= alpha1 <= 1
- featur2     is the second feature matrix
- alpha2      is the weight for the second feature matrix, with 0 <= alpha2 <= 1 and 0 <= alpha1+alpha2 <= 1
- Net         is the last feature matrix, which describes the relations among instances, its weight is 1-alpha1-alpha2
- beta        is the small value threshold
- num_paths   is the number of feature walks to start at each instance
- path_length is the length of the feature walk started at each instance
- dim         is the dimension of embedding representations
- win_size    is the window size of skipgram model

## Reference in BibTeX:
@conference{Huang-etal19Large,  
Author = {Xiao Huang and Qingquan Song and Fan Yang and Xia Hu},  
Booktitle = {AAAI Conference on Artificial Intelligence},   
Title = {Large-Scale Heterogeneous Feature Embedding},  
Year = {2019}}

