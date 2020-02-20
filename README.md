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
