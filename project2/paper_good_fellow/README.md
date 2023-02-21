## Model

The model is based on the [paper](https://www.cinc.org/archives/2017/pdf/361-352.pdf) by Sebastian D. Goodfellow, Andrew Goodwin, Robert Greer, Peter C. Laussen, Mjaye Mazwi1, and Danny Eytan. In contrast to the original implementation, we reimplemented the model in pytorch and trained it without weighing the individual labels since our model was evaluated using the f1 micro score. 

## How to run:

1. First run the train_classifier.ipynb file. The best model will be saved in "../models/". 
2. Take the path to the best model and set the variable "current_best_model" in create_features.ipynb to that value. Now run the create_features.ipynb notebook. 

Note that this folder only contains the jupyter-notebooks used to generate the features for fold 0. In order to create the features for fold 1, 2, 3, and 4, one needs to modify the current_fold variable in the train_classifier.ipynb and create_features.ipynb files. 

## Comment:

Due to time constraints, we were unable to finetune the models properly and just selected the best model after 100 epochs. We expect that with proper fine tuning one could achieve better results. Furthermore, adding some lstm layers might also lead to an improvement.
