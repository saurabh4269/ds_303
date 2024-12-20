# DS 303

This repository contains the solution for the **DS 303**, which covers the implementation and understanding of various machine learning algorithms, including Random Forests, AdaBoost, and cross-validation techniques.

## Table of Contents

- [Question 1: Random Forest](#question-1-random-forest)
  - [Part 1: Implement Random Forest functions](#part-1-implement-random-forest-functions)
  - [Part 2: Use Scikit Learn for Random Forest](#part-2-use-scikit-learn-for-random-forest)
  - [Part 3: Debugging Buggy Code](#part-3-debugging-buggy-code)
  - [Extra Credit: Cross Validation](#extra-credit-cross-validation)
  
- [Question 2: AdaBoost](#question-2-adaboost)
  - [Part 1: Implement AdaBoost functions](#part-1-implement-adaboost-functions)
  - [Part 2: Use Scikit Learn for AdaBoost](#part-2-use-scikit-learn-for-adaboost)
  
- [Extra Credit Question: Cross-Validation in Random Forests](#extra-credit-question-cross-validation-in-random-forests)

## Question 1: Random Forest

### Part 1: Implement Random Forest functions

The task was to implement the following functions in the `RandomForest.ipnb` file:
1. **Entropy Calculation**: Implement the function for entropy computation.
2. **RandomForestClassifier.fit**: Implement the function to train the random forest model.
3. **RandomForestClassifier.predict**: Implement the function to predict using the trained random forest model.


### Part 2: Use Scikit Learn for Random Forest

This part required using the standard `Scikit Learn` library to calculate the accuracy of the Random Forest classifier using the same dataset. The implementation uses the pre-built `RandomForestClassifier` from Scikit Learn.

### Part 3: Debugging Buggy Code

In this part, you need to debug the code in the `buggy_code.py` file. The file contained logical errors, including an overfitting problem.

### Extra Credit: Cross Validation

#### Cross Validation with Random Forests

In this extra credit section, the goal was to determine the optimal number of decision trees for a Random Forest model through T3-fold cross-validation. The following tasks were covered:
- **How many random forests are trained?**
- **How many decision trees are trained?**
- **Cross-validation implementation for hyperparameter tuning**


## Question 2: AdaBoost

### Part 1: Implement AdaBoost functions

The task was to implement the following functions in the `adaboost.py` file:
1. **Adaboost.fit**: Implement the function to train the AdaBoost classifier.
2. **Adaboost.predict**: Implement the function to predict using the trained AdaBoost classifier.


### Part 2: Use Scikit Learn for AdaBoost

This part required using Scikit Learn's AdaBoost classifier to calculate the accuracy of the model on the same dataset.

## Extra Credit Question: Cross-Validation in Random Forests

This extra credit question covered how to implement cross-validation to tune hyperparameters of the Random Forest model, particularly focusing on:
- The number of random forests to train.
- The number of decision trees in each fold.
- Cross-validation implementation for selecting optimal hyperparameters using the `KFold` class from Scikit Learn.
