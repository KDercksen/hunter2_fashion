This directory contains standalone scripts.

### download_images.py
Used to download the images given by the URLs in the JSON files acquired from
Kaggle. Usage:

    $ python download_images.py /path/to/kaggle-file.json /path/to/output-dir

    # Example to download training images into ./data/train:
    $ python download_images.py ./data/train.json ./data/train

To download `validation` or `test`, substitute filename and output directory as
appropriate.

### create_labels.py
Used to create an easy to use set of CSV/NPY files with image labels etc.
Usage:

    $ python create_labels.py /path/to/kaggle-dir /path/to/output-dir

    # Example to create help files in ./data:
    $ python create_labels.py ./data/ ./data/

### multilabel_nn.py
Proof of concept for multilabel classification. We use a `sigmoid` layer and
`binary_crossentropy` as loss function.

### simple_xception.py
Script to train two dense layers on top of a frozen Xception pretrained on
ImageNet.

Arguments:

    --batch-size <int> (default 64)
    --epochs <int> (default 10)
    --train-steps <int> (default full training set)
    --save-filename <str> (required, filename to save model to)
    --create-submission (if supplied, create submission after training)
    --continue-from-chpt <str> (if supplied, continue training from this
                                checkpoint filename)
    --windows (if supplied, disable multiprocessing)

### prediction_transformer.py
Script to fit models on network predictions.

Arguments:

    network-filename (required, filename of the network to load from `models`
                      directory)
    --epochs <int> (default 200)
    --train-steps <int>
    --batch-size <int> (default 128)
    --save-filename <str> (required, filename to save model to)
    --create-submission (if supplied, create submission)
