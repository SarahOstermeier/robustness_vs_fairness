# Analysis on the effect of robustness on the fairness using machine learning models.
## Authors 
Sarah Ostermeier/ Yukiko Suzuki

## Problem and technique description
#### What is the problem you're trying to solve?
It has been shown that differential privacy model affects the accuracy for underepresented groups more severely than the majority groups. We would like to test wheather the same affect applies if we use adversarial models to enhance robustness.
#### What have people done about it and what are the limitations of existing techniques?
https://arxiv.org/pdf/1907.00020.pdf
This paper tries to enhance fairness of the models through improving robustness. (The robustness is achieved regardless of subgroups of the dataset)

#### What do you propose to do?
We will train our base line models and equivalent adversarial models and evaluate the effect of the accuracies of each subgroup(majority/minority)

#### What are your measures of success?

---
Identify differences in accuracies in diffrerent subgroups.
If there is a difference, we will come up with techniques for mitigating the fairness problems.
In this case, our measure of success would be how much improvement we can bring to our initial adversarial models.
## Week of Feb/16th

#### On a scale of 1-10, how do we rate our progress over the past week?
10 
#### What did we accomplish from last week's tasks?
We came up with a project idea and defined an objective.
#### What problems or concerns do we have?
Getting started with implementations
#### What do we plan to accomplish do over the next week?
1. Find dataset
2. Set up AWS/Google Cloud
3. Decide on baseline models

## Week of Feb 23rd

#### On a scale of 1-10, how do we rate our progress over the past week?
10
#### What did we accomplish from last week's tasks?
We found a dataset of face images to use for our models, and decided to use Resnet as our base model.  Resnet can be imported pretrained
from Pytorch, so we plan to use this as our implementation approach.  We also found an example implementation of Resnet with adversarial
training, which we can use to guide our approach.

#### What problems or concerns do we have?
We are concerned about figuring out how to run our models on AWS or google cloud.

#### What do we plan to accomplish do over the next week?
We will figure out how to run our baseline model on either AWS or google cloud, and ensure that we can get it working.  If we have time, 
we will also begin to put together our adversarial model.

## Week of March 1st

#### On a scale of 1-10, how do we rate our progress over the past week?
5
#### What did we accomplish from last week's tasks?
We've spent time trying to set up AWS and google cloud, but have been running into problems, which we were not able to resolve given our time limitations over the past week

#### What problems or concerns do we have?
We are concerned about figuring out how to run our models on AWS or google cloud.

#### What do we plan to accomplish do over the next week?
We've decided to split our work.  One of us will figure out logistics of setting up a cloud environment for our project, while the other begins work on a toy version of our models.  

## Week of March 8th

#### On a scale of 1-10, how do we rate our progress over the past week?
10
#### What did we accomplish from last week's tasks?
Got virtual machine running on google cloud. Have begun work on base and adversarial models.  

#### What problems or concerns do we have?
No real concerns at this time

#### What do we plan to accomplish do over the next week?
We will continue to develop our models and test them over the next week 


## Week of March 15th

#### On a scale of 1-10, how do we rate our progress over the past week?
10
#### What did we accomplish from last week's tasks?
We got our first training running on the baseline and robust models

#### What problems or concerns do we have?
Somehow, the out of sample accuracy improved after running our adversarial models. There might be some bugs in our code. 

#### What do we plan to accomplish do over the next week?
Check where the bugs could lie
Create imbalanced dataset based on race, and run our models to see if the performance changes

## Mid-March Report
We successfully trained a baseline Resnet and an adversarially trained model on a gender classification task on the UKTface dataset.  After training, we compared test accuracy for each race subgroup.  Although we had predicted that a more robust (adversarially trianed) model may decrease accuracy in smaller subgroups, this initial test indicates otherwise.  In fact, our adversarially trained model performed with a higher accuracy in all subgroups compared to the baseline model.  To further investigate this result, we plan to retrain with an imbalanced dataset, such that each subgroup is about 5% the size of the majority group.  We will also train models to predict age and test under the same conditions.  

## Week of April 6th

#### On a scale of 1-10, how do we rate our progress over the past week?
10
#### What did we accomplish from last week's tasks?
We tried our model on the imbalanced dataset with 10000 white and 500 black

#### What problems or concerns do we have?
Somehow again, the fairness improved after the adversarial training for some of the minority groups
| VS                | White vs Black | White vs Asian | White vs Indian | White vs Others |
|-------------------|----------------|----------------|-----------------|-----------------|
| standard          | 79% vs 74%     | 81%  vs 69%    | 80% vs 81%      | 79% vs 77%      |
| adversarial (FSG) | 67% vs 53%     | 67% vs 70%     | 68% vs 71%      | 60% vs 58%      |


#### What do we plan to accomplish do over the next week?
Try with other minority races
Think about the reasons why adversarial training improved the fairness.
Try with different types attacks


## Literature Reviews

### Adversarial Training Can Hurt Generalization
https://openreview.net/pdf?id=SyxM3J256E
The adversarily trianed model improves robust accuracy but may worsen the standard accuracy. The authors argue that this is due to lack of generalization and maybe solved by increasing the sample size

### ENSEMBLE ADVERSARIAL TRAINING: ATTACKS AND DEFENSES
https://arxiv.org/pdf/1705.07204.pdf
The model we use for our adversarial training. It uses *static* learned model to generate adverasarial samples. 

### Mitigating Unwanted Biases with Adversarial Learning
http://m-mitchell.com/papers/Adversarial_Bias_Mitigation.pdf
Uses a method called adversarial debiasing to train a deep learning model to accurately predict an output Y, given input X, while remaining unbiased with respect to variable Z (the "protected variable")
=======
### https://www.fatml.org/media/documents/achieving_fairness_through_adversearial_learning.pdf
https://www.fatml.org/media/documents/achieving_fairness_through_adversearial_learning.pdf
Uses adversarial training to improve fairness. Assume that we have some subset of dataset with race attribute known.
We train 2 models simultaneouly which try to predict the label(gender) and the race attribute.
$f(g(x)) = y $
$a(g(x)) = z$
where y is the label (gender) and z is the partially hidden attribute(race). The goal is for f and a to predict y and z accurately. But we also try to make g harder for a to predict z. Note that g is the embedding layer used for both functions.

### Usefull Links:
Face Dataset:  https://www.kaggle.com/jangedoo/utkface-new

Resnet Pretrained Pytorch:
https://pytorch.org/docs/stable/torchvision/models.html

Resnet with Adversarial training example
https://github.com/JZ-LIANG/Ensemble-Adversarial-Training

How to run jupyter notebooks on google cloud
https://tudip.com/blog-post/run-jupyter-notebook-on-google-cloud-platform/

