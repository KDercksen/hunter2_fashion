iMaterialist Challenge (Fashion) at FGVC5
=========================================

This repository hosts our code for the [iMaterialist Fashion
Challenge](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018).

### Setup
Install Python 3.6 or higher and all dependencies necessary in a virtual
environment. Then install this project with the editable option:

    $ pip install -e .

Download the Kaggle data files into the `data` directory, and run the following
scripts (includes deletion of broken images):

    $ python scripts/download_images.py ./data/train.json ./data/train
    $ python scripts/download_images.py ./data/validation.json ./data/validation
    $ python scripts/download_images.py ./data/test.json ./data/test
    $ python scripts/delete_broken_images.py

    $ python scripts/create_labels.py ./data/ ./data/

All scripts should be ran from the project root directory, e.g.:

    $ python scripts/multilabel_nn.py

This is necessary to make the relative imports and path strings work properly.