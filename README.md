EmotionEstimation
==============================

This is a project to estimate emotions of PIMD from facial images.

Project Organization
------------

    ├── LICENSE
    |
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    |
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── data
    |   |
    │   ├── external       <- Data from third party sources.
    |   |
    │   ├── interim        <- Intermediate data that has been transformed, i.e. transformed imgs.
    |   |
    |   ├── labels         <- Labels for all data.
    |   |
    |   ├── params         <- Trained weights of third party models and snapshots of ours.
    |   |
    │   ├── processed      <- The final, canonical data sets for modeling, i.e. features.
    |   |
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs(NOT USE)      <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Third party models downloaded from github.(except weights)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- The result of experiments.
    |
    ├── runs               <- bash files to run training and inference(test).
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── src                <- Source code for use in this project. Naming convention is a verb, a 
    |   |                     short `-`, and delimited description, e.g. `train-model.py`.
    |   |
    │   ├── networks       <- Scripts to define each network.
    |   |
    |   ├── notebooks      <- Jupyter notebooks.  
    |   |
    │   ├── utils          <- Scripts to use each situation.
    |   |
    |   ├── test-model.py  <- Script to test model. Head number indicates the version.
    |   |
    |   ├── train-model.py <- Script to train model. Head number indicates the version.
    |   |
    |   ├── dataset.py     <- Script to set up dataset.
    |   |
    │   └── preprocess.py  <- Script to set up images preprocess.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
