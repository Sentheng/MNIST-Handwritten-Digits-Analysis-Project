function [accuracy, Ytest] = KNNfunction(KNNk, testLabels, RtestImages, RtrainImages, trainLabels)
%KNN Algorithm using a for loop to iterate through the different values of k 
for k = KNNk
    Ytest = zeros(size(testLabels)); % Setting the size of the Predicted matrix to size of test Labels
    for i = 1:size(RtestImages, 1) % This will be looping through all of the reshaped test Images
        
        % Compute distances between testing data and the training data using the Eucledean distance
        %Distance between the neighbors to determine class of example
        dist = sqrt(sum((RtrainImages - repmat(RtestImages(i,:), size(RtrainImages,1), 1)).^2, 2));
        
        % Creating a variable that sorted index from the sorted 
        [sortedD, index] = sort(dist);
        
        % Select k-nearest neighbors
        knnLabels = trainLabels(index(1:k));
        
        % Determine class of test image
        Ytest(i) = mode(knnLabels);
    end
    
    % Compute accuracy of k-nearest neighbor
    accuracy = sum(Ytest == testLabels) / length(testLabels);
    fprintf('k = %d: %.4f\n', k, accuracy); % Accuracy is probability  
end 

end