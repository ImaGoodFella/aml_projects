# Project 2

## Task Description

This task is primarily concerned with time-series classification. We are given ECG signals, which we must classify into four groups.

## Dataset

This is a supervised learning task. We are given a training dataset (X_train, y_train) and a test dataset (X_test). The objective is to determine the corresponding y_test for the X_test dataset. Due to privacy concerns, I am not allowed to upload the datasets. My code assumes that they are in the current folder. 

## How to run

1. Run the manual_feature_extractor.ipynb notebook
2. Run the create_folds.ipynb notebook in the split folder
3. Follow the instructions in the resnet_model folder
4. Follow the instructions in the paper_good_fellow folder
5. Run the voting_predictor.ipynb notebook

## Model

The final prediction is a hard classification of 5 estimators. Each estimator is trained with features extracted manually and features extracted by neural networks trained on a different fold of the data. 

The estimator is a Final StackingClassifier with LGBM, XGBoost, RandomForest and HistGradient classifiers.

### Manual feature extraction

We use neurokit2 and biospy2 to extract the P, Q, R, S, T peaks and compute their relative positions, amplitudes, intervals, and relations. We added some statistical measures on them as well.
We use the libraries hrv-analysis and heartpy to get the heart rate variability and other non-linear features. 
We also extract some FFT features.

### Neural network feature extraction

We implemented the neural networks described in papers for the Physionet challenge. We modified the training procedure and models to adjust for the difference between the Physionet challenge and our task. Further information on how to train the models can be found in the respective folder.

#### Resnet feature extraction

We use a ResNet model based on "ENCASE: an ENsemble ClASsifiEr for ECG Classification Using Expert Features and Deep Neural Networks";. The model has 48 convolutional blocks, where each block consists of two 1d convolutional, batch norm, max pool and drop-out layers. We extract 32 features from the last layer.

#### Goodfellow feature extraction

We implemented the model described in "Towards Understanding ECG Rhythm Classification Using Convolutional Neural Networks and Attention Mappings". We extract the 64 features from the last layer. 

## Comments
This task is very similar to the [Physionet 2017](https://physionet.org/content/challenge-2017/1.0.0/) challenge. However, the evaluation is slightly different, with the Physionet challenge using an f1 macro score to evaluate the models, whereas the AML project evaluates the model using an f1 micro score. This leads to a different training procedure for the neural networks and models since class imbalance becomes almost irrelevant when evaluating the model using the f1 micro score. 

