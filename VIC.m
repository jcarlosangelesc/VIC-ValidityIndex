close all
clear all
clc

classifiers = ["LDA", "DecisionTree", "RandomForest", "KNN"];
kFolds = 10;
partitions = 50;

evaluateClassifiers(classifiers, kFolds, partitions);