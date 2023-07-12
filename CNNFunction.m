%CNN

function [accuracy, YPred] = CNNFunction( testLabels, CNNImagesTrain, CNNImagesTest, trainLabels)

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 20) % do i change this 
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];



% Display the network
%analyzeNetwork(layers);

%Training Aspect
% options = trainingOptions('sgdm', ...
% 'InitialLearnRate',0.01, ...
% 'MaxEpochs',4, ...
% 'Shuffle','every-epoch', ...
% 'ValidationData',trainImages, ...
% 'ValidationFrequency',30, ...
% 'Verbose',false, ...
% 'Plots','training-progress');

options = trainingOptions('sgdm', 'MaxEpochs', 10, 'InitialLearnRate', 0.01);

%Testing Aspect
net = trainNetwork(CNNImagesTrain, categorical(trainLabels),layers,options);

%Calculating Accuracy
YPred = classify(net,CNNImagesTest);

accuracy = sum(YPred == categorical(testLabels))/numel(testLabels);

fprintf('Accuracy for CNN: %.4\n', accuracy);
end 
