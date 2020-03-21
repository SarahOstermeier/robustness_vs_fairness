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

### Usefull Links:
Face Dataset:  https://www.kaggle.com/jangedoo/utkface-new

Resnet Pretrained Pytorch:
https://pytorch.org/docs/stable/torchvision/models.html

Resnet with Adversarial training example
https://github.com/JZ-LIANG/Ensemble-Adversarial-Training

How to run jupyter notebooks on google cloud
https://tudip.com/blog-post/run-jupyter-notebook-on-google-cloud-platform/

