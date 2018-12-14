# Sensitivity of deep learning facial recognition models to adversarial inputs
Fall 2018-2019

* `noise.py` contains our scripts for adding noise to images
* `merge_training_set.py` is used to merge noisy and non-noisy inputs for adversarial training
* `limit_training_set.py` is used to limit the number of training images used to balance images per class
* `facial_landmarks/landmarks_nn.py` stores our DNN used to perturb facial landmarks; it can be trained on a laptop
