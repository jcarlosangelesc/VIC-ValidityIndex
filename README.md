# VIC-ValidityIndex
VIC [1] is a Cluster Validation technique that uses a set of supervised classifiers to select which clustering algorithm to apply for a given problem.

This implementation was developed in MATLAB, and uses a custom dataset of human fingerprints represented as minutiae for fingerprint recognition. For more information on the data set, please check `THE REPORT HERE.pdf`.

VIC uses k-folds cross validation to train the ensemble of supervised classifiers on each partition of the dataset. Once the classifiers are trained, the mean ROC AUC of all the k-folds is used as an indicator of the validity of each classifier. In the end, the highest AUC value among all partitions indicates the most optimal partition of the data. The algorithm works as follows:


    INPUT dataset, classifiers, kFolds, nPartitions;
    OUTPUT report;  

    report = [];
    features, partitions = splitDataset(dataset, partitions);

    FOR EACH p IN nPartitions:
        partition = partitions[p];
        classifierMeans = [];
        FOR EACH c IN classifiers:
            AUCs = [];
            FOR EACH k IN kFolds:
                trainX, trainY, testX, testY = kFoldSplit(features, partition);
                trainClassifier(c, trainX, trainY);
                AUCs[k] = calculateAUC(c, testX, testY);
            meanAUC = mean(AUCs);
            classifierMeans[c] = meanAUC;
        report[p] = [p, classifierMeans, max(classifierMeans)];

## Available Classifiers

The current implementation supports 6 classifiers:

1. Random Forest 
2. K-Nearest Neighbors
3. Decision Trees
4. Linear Discriminant Analysis
5. Naive Bayes
6. Support Vector Machine

## Adding/Removing Classifiers

Adding new classifiers is as simple as adding an `elseif` statement in the `if/else` block starting from line 59 on `evaluateClassifiers.m` following the existing examples present. For example:

    ...
    elseif strcmp(classifiers(c), 'NewClassifier')
        classifier = fitclassifier(trainX, trainY, parameter1, ..., parameterN);
    ...

Please notice that the new classifier must have the methods `fitclassifier(X, Y)` and `predict(X)` for model training and testing, respectively. For MATLAB default classifiers this is not an issue as all of them follow this convention, however, for custom classifiers please make sure that this is the case.

The final step is to add or remove classifiers as desired from the `classifiers` array when calling `evaluateClassifiers`.

## Usage

To test the current implementation all you need to is:

1. Open `VIC.m` in MATLAB.
2. Define the classifiers to use by adding or removing classifiers from the `classifiers` array.
3. Specify the number of folds for k-folds cross validation.
4. Run the script.
5. Select the csv dataset when prompted.

The script will generate a text file named `evaluation.txt` with the results for all the partitions. **By default, MATLAB functions automatically execute on multiple computational threads since version 2008a.**

When running on a different dataset, make sure to add a column for all the different partitions that you want to evaluate on the rightmost side of your dataset as follows:

Feature 1 | ... | Feature N | Partition 1 | ... | Partition P |
--------- | --- | --------- | ----------- | --- | ----------- |
... | ... | ... | ... | ... | ... |

The algorithm will take care of splitting the partitions and choosing the right one every iteration.

## Authors

* Laura Pérez - ([https://github.com/LauraJaideny](https://github.com/LauraJaideny))
* Andreé Vela - ([https://github.com/AndreeVela/](https://github.com/AndreeVela/))
* Juan Carlos Ángeles - ([https://github.com/jcarlosangelesc](https://github.com/jcarlosangelesc))

## Bibliography

[1] J. Rodríguez, M. A. Medina-Pérez, A. E. Gutierrez-Rodríguez, R. Monroy, H. Terashima-Marín. [Cluster validation using an ensemble of supervised classifiers](https://www.sciencedirect.com/science/article/abs/pii/S0950705118300091). *Knowledge-Based Systems*, Volume 145 (2018). Pages 134-144.
