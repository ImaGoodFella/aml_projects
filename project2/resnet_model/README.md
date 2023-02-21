## Model

The model is based on the [github](https://github.com/hsd1503/resnet1d) implementation. We first create a resnet model with a simple linear classifier. In contrast to the original implementation, we then remove the linear layer and a classifier with 2 linear layers which we then use to extract 32 features. Furthermore, we trained the model without weighing the individual labels since our model was evaluated using the f1 micro score. 

## How to run:

1. Run the train_classifier.ipynb notebook up to the first pytorch training loop. The best resnet classifier will be saved in "../models/". 
2. Take the path to the best linear classifier and set the "current_best" variable in train_classifier.ipynb to that value. Now completely run the notebook. The best resnet extractor will be saved in "../models/"
3. Take the path to the resnet extractor and set the variable "current_best_model" in create_features.ipynb to that value. Now run the create_features.ipynb notebook. 

Note that this folder only contains the jupyter-notebooks used to generate the features for fold 0. In order to create the features for fold 1, 2, 3, and 4, one needs to modify the current_fold variable in the train_classifier.ipynb and create_features.ipynb files. 

## Comment:

Due to time constraints, we were unable to finetune the models properly and just selected the best model after 2 times 50 epochs. We expect that with proper fine tuning one could achieve better results. Furthermore, adding some lstm layers might also lead to an improvement.
