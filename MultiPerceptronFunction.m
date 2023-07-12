%Multilayer Perceptron
function [accuracy, Ytest] = MultiPerceptronFunction(hiddenNeurons, testLabels, RtestImages, RtrainImages, trainLabels)
neurons = [50,100];

for a = 1:hiddenNeurons % a is the index of the network of neurons
    val = neurons(a);
     mlpTest = fitcnet(RtrainImages,trainLabels,'LayerSizes',val); % Running MLP
    [Ytest,Score]=predict(mlpTest,RtestImages); %Prediction Aspect
    
     accuracy = sum(Ytest == testLabels) / length(testLabels); 
    fprintf(' %d neurons: %.4f\n\n', val, accuracy); %Accuracy is a probability
    
end
end