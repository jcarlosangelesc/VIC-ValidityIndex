function [] = evaluateClassifiers(classifiers, kFolds, nPartitions)

% Prompt the user for csv file.
[filename, pathname] = uigetfile('*.csv');

if filename == 0
    disp('Operation cancelled');
else
    tic
    
    % Create the output file and write header.
    outputFile = fopen('evaluation.txt','w');
    fprintf(outputFile, '%s,', ["Partition", classifiers, "Maximum"]);
    fprintf(outputFile, '\n');

    % Read the data from the csv table
    data = readtable(strcat(pathname, filename));

    % Separate feature columns from partition columns
    features = data(:, 1:end-nPartitions);
    features = fillmissing(features, 'nearest');
    partitions = table2cell(data(:, end-nPartitions+1:end));

    % Loop through each partition and calculate AUCs for all the specified
    % classifiers
    for p=1:nPartitions
        disp(['Analyzing partition ', int2str(p)])
        partition = partitions(:, p);
        info = [strcat("Partition ", int2str(p))];
        classifierMeans = zeros(1, length(classifiers));

        % Split the dataset into k partitions for cross validation
        folds = cvpartition(partition,'k', kFolds);
        AUCs = zeros(1, kFolds);

        % Loop through all the specified classifiers
        for c=1:length(classifiers)
            disp(sprintf('Classifier: %s', classifiers(c)));
            % Cross validation
            for k = 1:kFolds
                foldTime = toc;
                % Train and test split for the current fold
                trainIdx = folds.training(k);
                testIdx = folds.test(k);
                valid = true;

                % Here are all the available classifiers. You may change
                % their parameters according to your needs.
                if strcmp(classifiers(c), 'RandomForest')
                    classifier = TreeBagger(100, features(trainIdx, :), partition(trainIdx, :), 'Method', 'classification');
                elseif strcmp(classifiers(c), 'KNN')
                    classifier = fitcknn(features(trainIdx, :), partition(trainIdx, :),'NumNeighbors',5,'Standardize',1);
                elseif strcmp(classifiers(c), 'DecisionTree')
                    classifier = fitctree(features(trainIdx, :), partition(trainIdx, :));
                elseif strcmp(classifiers(c), 'LDA')
                    classifier = fitcdiscr(features(trainIdx, :), partition(trainIdx, :), 'discrimType', 'pseudoLinear');
                elseif strcmp(classifiers(c), 'NaiveBayes')
                    classifier = fitcnb(features(trainIdx, :), partition(trainIdx, :));
                elseif strcmp(classifiers(c), 'SVM')
                    classifier = fitcecoc(features(trainIdx, :), partition(trainIdx, :));
                else
                    disp('Unknown classifier')
                    continue
                end

                if valid
                    % Train the classifier and calculate the ROC AUC for
                    % this fold and store the result.
                    [Yfit,scores,stdevs] = classifier.predict(features(testIdx, :));
                    [X,Y,T,AUC] = perfcurve(partition(testIdx, :),scores(:,1),'+');
                    AUCs(k) = AUC;
                    disp(sprintf('Cross Validation: %d AUC: %3.2f Time: %3.4f', k, AUC, toc-foldTime));
                end
            end
            
            % Compute the mean AUC of the current classifier and store the
            % result.
            meanAUC = mean(AUCs);
            disp(sprintf('Mean AUC: %3.4f \n', meanAUC));
            classifierMeans(c) = meanAUC;
            info = [info, meanAUC];
        end
        % Write the results of all the classifiers to the output file
        info = [info, max(classifierMeans)];
        fprintf(outputFile, '%s,', info);
        fprintf(outputFile, '\n');
        disp('')
    end
 
    disp(sprintf('Total time of execution: %3.4f', toc));
    fclose(outputFile);
end
end

