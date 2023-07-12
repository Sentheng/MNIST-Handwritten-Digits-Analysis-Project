function [accuracy, Ytest] = KMeanFunction(KMEANSk, testLabels, RtestImages, RtrainImages, trainLabels)







%KMeans Algorithm using a for loop to iterate through the different values of k 
for k1 = KMEANSk
    % Centriods assignment(This is done randomly,thus we need to use randperm)
    trainingSize = size(RtrainImages,1);
    IndexRandom = randperm(trainingSize, k1);
    cent = RtrainImages(IndexRandom, :);


    for i = 1:15 % 1 to 15 iterations
        % Assignment Stage for each point 
        dist = pdist2(RtrainImages, cent);
        [~, IndexCluster] = min(dist, [], 2); % Assigning to a specific cluster

        % Updating Stage for each cluster means as each new point is added
        for j = 1:k1
            cluster1 = RtrainImages(IndexCluster == j, :);
            cent(j, :) = mean(cluster1, 1);
        end
    end

    % Classify test data based on closest centroid
    dist = pdist2(RtestImages, cent);
    [~, Ytest] = min(dist, [], 2);

    % Compute accuracy of k-means classification
    accuracy = sum(Ytest == testLabels) / length(testLabels);
    fprintf('k = %d: %.4f\n', k1, accuracy); % Accuracy is probability  
end

%Confusion Matrix
confusion_matrix = zeros(length(unique(testLabels))); % Setting the confusion matrix

confusion_matrix_k = confusionmat(testLabels, Ytest); % Creating the matrix
confusion_matrix = confusion_matrix + confusion_matrix_k;

disp(confusion_matrix);

end