close all
clear all
clc

classifiers = ["LDA", "DecisionTree", "RandomForest", "KNN"];
evaluateClassifiers(classifiers, 10, 50)