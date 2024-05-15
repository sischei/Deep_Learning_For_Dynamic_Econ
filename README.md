# Deep Learning for Solving Dynamic Stochastic Models (May 22 – 24, 2024)

This is an mini-course on "Deep Learning for Solving Dynamic Stochastic Models", held from Wednesday, May 22nd, 2024 2 - Friday, May 24th, 2024 at [Central-German Doctoral Program Economics, University of Leipzig](http://cgde.wifa.uni-leipzig.de/).


## Purpose of the lectures

* This mini-course is designed for Ph.D. students in economics and related disciplines. It introduces recent advancements in applied mathematics, machine learning, computational science, and the computational economics literature. The course focuses on solving and estimating dynamic stochastic economic models and performing parametric uncertainty quantification.

* The lectures will concentrate on two machine learning methodologies: Deep Neural Networks and Gaussian Processes. These methods will be explored through applications in macroeconomics and climate-change economics.

* The format of the lectures will be interactive and workshop-like, combining theoretical discussions with hands-on coding exercises. The coding will be conducted in Python and implemented on a cloud computing infrastructure.


## Class enrollment on the [Nuvolos Cloud](https://nuvolos.cloud/)

* All lecture materials (slides, codes, and further readings) will be distributed via the [Nuvolos Cloud](https://nuvolos.cloud/).
* To enroll in this class, please click on this [enrollment key](https://app.nuvolos.cloud/enroll/class/OW-jhN1vUjU), and follow the steps.


### Novolos Support

- Nuvolos Support: <support@nuvolos.cloud>


## Prerequisites

* Basic econometrics.
* Basic programming in Python (see [this link to QuantEcon](https://python-programming.quantecon.org/intro.html) for a thorough introduction).
* A brief Python refresher is provided [under this link](python_refresher).
* A brief Python on Jupyter Notebooks is provided [under this link](python_refresher/jupyter_intro.ipynb) 
* Basic calculus and probability (The book [Mathematics for Machine learning](https://mml-book.github.io/) provides a good overview of skills participants are required to be fluent in). 


## Topics

### [Day 1](lectures/day1), Wednesday, May 22nd, 2024 

 **Time** | **Main Topics** 
------|------
09:30 - 11:00 | [Introduction to Machine Learning and Deep Learning](lectures/day1/slides/01_Intro_to_DeepLearning.pdf) (2 x 45 min)
11:00 - 11:30 | Coffee Break
11:30 - 13:00 | [A hands-on session on Deep Learning, Tensorflow, and Tensorboard](lectures/day1/code) (2 x 45 min)
13:00 - 14:30 | Lunch Break 
14:30 - 16:00 | [Introduction to Deep Equilibrium Nets (DEQN)](lectures/day1/slides/02_DeepEquilibriumNets.pdf) (2 x 45 min)

### [Day 2](lectures/day2), Thursday, May 23nd, 2024 

 **Time** | **Main Topics** 
------|------
09:30 - 10:15 | Hands-on: Solving a stochastic dynamic models by [example](lectures/day2/code/02_Brock_Mirman_Uncertainty_DEQN.ipynb)  (45 min)
10:15 - 10:45 | Exercise: Solving a stochastic dynamic models by [example](lectures/day2/code/03_DEQN_Exercises_Blancs.ipynb)  (45 min)
11:00 - 11:30 | Coffee Break
11:30 - 12:15 | Exercise: Solving a stochastic dynamic models by [example](lectures/day2/code/03_DEQN_Exercises_Blancs.ipynb)  (45 min)
12:15 - 13:00 | [Introduction to the DEQN library](lectures/day2/code/DEQN_production_code): [solving a stochastic dynamic OLG model with an analytical solution](lectures/day2/slides/OLG_with_analytical_solution.pdf) (45 min)
13:00 - 14:30 | Lunch Break 
14:30 - 16:00 | [Surrogate models part I](lectures/day2/slides/01_Surrogate_models.pdf) (for structural estimation and uncertainty quantification via [Gaussian process regression](lectures/day2/readings/Machine_learning_dynamic_econ.pdf) and [deep surrogate models](lectures/day2/readings/Deep_Surrogates.pdf)) (2 x 45 min)

### [Day 3](lectures/day3), Friday, May 24th, 2024

 **Time** | **Main Topics** 
------|------
09:30 - 10:15 | [Surrogate models part II](lectures/day2/slides/01_Surrogate_models.pdf)
10:15 - 11:00 | Introduction to integrated assessment models (45 min)
11:00 - 11:30 | Coffee Break
11:30 - 13:00 | Solving the (non-stationary) DICE model with Deep Equilibrium Nets (2 x 45 min)
13:00 - 14:30 | Lunch Break 
14:30 - 16:00 | Putting things together: [Deep Uncertainty Quantification for stochastic integrated assessment models]; wrap-up of course (2 x 45 min)


### Teaching philosophy
Lectures will be interactive, in a workshop-like style,
using [Python](http://www.python.org), [scikit learn](https://scikit-learn.org/), [Tensorflow](https://www.tensorflow.org/), and
[Tensorflow probability](https://www.tensorflow.org/probability) on [Nuvolos](http://nuvolos.cloud),
a browser-based cloud infrastructure in which files, datasets, code, and applications work together,
in order to directly implement and experiment with the introduced methods and algorithms.


### Lecturer
- [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (HEC, University of Lausanne)


# Auxiliary materials 

| Session #        |  Title     | Screencast  |
|:-------------: |:-------------:| :-----:|
|   1 	|First steps on Nuvolos | <iframe src="https://player.vimeo.com/video/513310246" width="640" height="400" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>|
|   2 	| Terminal intro | <iframe src="https://player.vimeo.com/video/516691661" width="640" height="400" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>|
