# Chapter 1

***

### 1.1: A Motivating Example

*Machine learning* is all about *coding with data* - instead of hard coding the cat detector, we can feed our machine data, and let it *learn* how to distinguish cats from dogs.

***

### 1.2 Key Components

Our problem from the first chapter, in which we try to **predict a designated unknown label based on known inputs given a dataset consisting of examples for which the labels are known** is called *supervised learning*.

Here a key components of every machine learning problem, regardless of it's type:

1. The *data* we can learn from.
2. A *model* of how to transform the data.
3. An *objective function* that quantifies how well the *model* is doing.
4. An *algorithm* to adjust the model's parameters to optimize the *objective function*.

#### 1.2.1: Data

Each *example* (or *data point*, *data instance* or *sample*) typically consists of a set of attributes called *features*, based on which the model must make its predictions.

In *supervised learning*, our goal is to predict the value of the *label*, which is not part of our models input.

In cases in which every example is characterized by the same number of numerical features, we call the length of the vectors the *dimensionality* of the data. Deep learning enables us to handle *varying-length data*.

#### 1.2.2: Models

By *model*, we denote the machinery for ingesting data of one type, and spitting out predictions of a possibly different data type. In particular, we are interested in *statistical models* that can be estimated from data.

#### 1.2.3: Objective Functions

We introduced machine learning as learning from experience. By *learning*, we mean *improving* at some task over time. In optimization, an *objective function* is a formal measure of how good our *models* are. By convention, lower is better, that's why we also sometimes call objective functions *loss functions*.

The most common loss function for predicting numerical values is *squared error*, i.e., the square of the difference between the predictions and the ground truth targets. For classification tasks, the most common objective is to minimize *error rate*, so to say the fraction of examples on which our model disagrees with the ground truth. These are typically difficult to optimize directly. 

Doing well on the training dataset is not enough: If we want to do well on *unseen* data, we want to split the available data into a *training dataset* and a *test dataset* to evaluate on afterwards. => Practice exam scores and final exam scores are both reported.

When a model performs well on the training set but fails to generalize to unseen data, we say that it's *overfitting* the training data.

#### 1.2.4: Optimization Algorithms

Once we have got a data source and representation, a model, and a objective function, we need an algorithm to search for the optimal set of *parameters* to minimize the loss function. => **Gradient descent**

***

### 1.3 Kinds of Machine Learning Problems

A broad overview of the landscape of machine learning problems.

#### 1.3.1: Supervised Learning

Given a dataset containing both *features* and *labels*, we try to produce a *model* that predicts the labels when given input features. Each feature-label-pair is called *example*.

In probabilistic terms, we typically are interested in estimating the conditional probability of a label given input features. => "predicting the labels given input features"

A broad overview of the learning process: We grab a collection of examples, selecting a random subset of them, and then acquiring the correct labels for each. Together, these inputs and labels comprise the training set, which we now feed into a learning algorithm, a function that takes as input a dataset and outputs the learned model function.

Typical supervised learning problems are:

1. *Regression* - Problems in which the label takes on arbitrary numerical values, for which we try to find a model that closely approximates the actual label values => **house pricing problem**, **"how much? how many? problems"** 
2. *Classification* - Pretty self-explanatory: our model should look at features, and then predict to which *category* among some *discrete* set of options an example belongs. Often times, we can't output a *firm* classification, but a probabilistic confidence of the models classification decision.
3. *Tagging* - Think of this like classification, but for multiple subjects at once. The problem of learning to predict classes that are not *mutually exclusive* is called *multi-label classification*.
4. Search - *PageRank* is a typical example of this problem. We don't try to only find out which results are relevant, but which results are the most relevant for a user.
5. Recommender Systems
6. Sequence Learning
7. Tagging and Parsing
8. Automatic Speech Recognition
9. Text to Speech
10. Machine Translation

#### 1.3.2: Unsupervised and Self-Supervised Learning

Think of supervised learning as having a boss and a learner, with the boss standing behind the learner and dictating him on the correct label for each feature set. Suppose the opposite situation - that's *unsupervised learning*. Here are some questions we might ask:

- Can we find a small number of prototypes that accurately summarize the data? => **Clustering**
- Can we find a small number of parameters that accurately capture the relevant properties of the data? => **subspace estimation**
- **Deep Generative Models** - These models estimate the density of the data, either explicitly or implicitly. Once trained, we can use a generative model either to score examples based off of likelihood, or to sample synthetic examples.

#### 1.3.3: Interacting with an Environment

Supervised learning and unsupervised learning approaches normally don't concern themselves with a connection to their environment, because all the learning typically takes place once the algorithm is disconnected from the environment, as it got it's input data.

In comparison, we want to think about intelligent *agents*, not just predictive models. This means that they need to choose *actions*, and not just predicting stuff. Considering the interaction with an environment opens a whole set of new modeling questions:

- Does the environment remember what we did previously?
- Does the environment want to help/beat us?
- Does the environment have shifting dynamics? Would future data always resemble the past, or would patterns change over time?

This can lead to *distribution shift*, where training and testing data are different.

#### 1.3.4: Reinforcement Learning

**Reinforcement learning** gives a very general statement of a problem in which an agent interacts with an environment over a series of time steps. At each time step, the agent receives some *observation* from the environment and must choose an *action* that is subsequently transmitted back to the environment, when, after each loop, the agent receives a *reward* from the environment.

The behavior of a reinforcement learning agent is governed by a *policy*. In brief, a *policy* is just a function that maps from observations of the environment to actions.
