%Nathan Theng 
%CSCI 164
%Main Test File for Project 2

close all % Clearing everything everytime it is runned
clc 
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Picking Algorithm %%%%%%%%%%%%%%%%%%%%%%%%

% If choice = 1, run KNN 
% If choice = 2, run Kmean 
% If choice = 3, run MultiPerceptionLayer
% If choice = 4, run CNN

choice = 4;  %% CHANGE ALGO HERE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Importing Test Data %%%%%%%%%%%%%%%%%%%%%%

nclasses = 10; %Classes in the data
nexamples = 500; % We want 500 examples from the classes
    
load("data_mnist_test_original.mat"); %Only requires the test dataset 
testImages = reshape(imgs, 28*28,[])';
testLabels = labels; %No need to reshape the labels. Only the Images
newTestImages = [];
newTestLabels = [];

for i = 1:nclasses %Looping through all the classes to obtain the 500... will get 5000 examples in total
    cindex = find(testLabels == (i-1));% We want to get to 0 not 1
    cindex = cindex(1:nexamples);

    %Creating new dataset/dataframe
    newTestImages = [newTestImages; testImages(cindex,:)];
    newTestLabels = [newTestLabels; testLabels(cindex)];
end

%Normalizing the Data
newTestImages = double(newTestImages)/255;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Importing Train Data %%%%%%%%%%%%%%%%%%%%%

load("data_mnist_train_original.mat") %Loading the training set
trainImages = reshape(imgs, 28*28,[])';
trainLabels = labels; %No need to reshape the labels. Only the Images
newTrainImages = [];
newTrainLabels = [];

for i = 1:nclasses %Looping through all the classes to obtain the 500 per class... will get 5000 examples in total
    cindex1 = find(trainLabels == (i-1));% We want to get to 0 not 1 and selecting the first 500
    cindex1 = cindex1(1:nexamples); 

    %Creating new dataset/dataframe
    newTrainImages = [newTrainImages; trainImages(cindex1,:)];
    newTrainLabels = [newTrainLabels; trainLabels(cindex1)];
end

%Normalizing the Data
newTrainImages = double(newTrainImages)/255;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KNN Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Different K values of KNN
KNNk = [1, 5, 10]; % Three values for testing the accuracy 

if choice == 1 

   fprintf('Accuracy Testing for KNN Algo: \n')

   for k = KNNk
        CM = zeros(nclasses,nclasses);
        Yout = zeros(size(newTestImages,1),1); % Setting the size of the Predicted matrix to size of test Labels
        for i = 1:size(newTestImages,1) % This will be looping through all of the reshaped test Images
            theoritcalEx = newTestImages(i,:);

            % Compute distances between testing data and the training data using the Eucledean distance
            %Distance between the neighbors to determine class of example
            dist = sqrt(sum((newTrainImages - theoritcalEx).^2, 2));
            
            % Creating a variable that sorted index from the sorted 
            [~, index] = sort(dist);
            
            % Select k-nearest neighbors
            knnLabels = newTrainLabels(index(1:k));
            
            % Determine class of test image
            Yout(i) = mode(knnLabels);

            val = newTestLabels(i);
            Ypred = Yout(i);
            CM(val + 1, Ypred +1) =  CM(val + 1, Ypred +1)+1;  

             % Compute accuracy of k-nearest neighbor
      
        end
        accuracy = sum(Yout == newTestLabels) /(size(newTestImages,1));
        fprintf('k = %d: %.4f%%\n', k, accuracy*100); % Accuracy is probability
        
        CMdisplay = zeros(nclasses+1,nclasses+1);
        CMdisplay(1,2:11)=0:9;
        CMdisplay(2:11,1)=0:9;
        CMdisplay(2:11,2:11)=CM;

        fprintf('CM')
        disp(CMdisplay);
        
    end 
        

end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kmeans Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%

KMEANSk = [10,20,30]; %Three values for testing the accuracy 
if choice == 2
    fprintf('Accuracy Testing for KMeans Algo: \n')
 
    for k = KMEANSk %Looping through the Kmeans and running Kmeans
      cent = newTestImages(randi(size(newTestImages,1),1,k), :); %Stage 1: Assignment centroids
        
        for i = 1:20
            dist = pdist2(newTestImages,cent);
            [~,testLabels] = min(dist,[],2);
            
            for j = 1:k  %Stage 2: Updating centroids
                cent(j,:) = mean(newTestImages(testLabels == j, :), 1);
            end
        end
     
        Yout = zeros(1,k);
        for a =1:k
            b = testLabels == a;
            l = mode(newTestLabels(b));
            Yout(a) = sum(newTestLabels(b) == l);
        end
    
        %Accuracy Testing
        accuracy = sum(Yout)/size(newTestImages,1);
        fprintf('k = %d: %.4f%%\n', k, accuracy*100); % Accuracy is probability 
    
        %Confusion Matrix
        CM = zeros(nclasses,nclasses);
    
        for m = 1:size(newTestImages,1);
            val = newTestLabels(m);
            Ypred = mode(newTestLabels(testLabels == testLabels(m)));
            CM(val + 1, Ypred +1) =  CM(val + 1, Ypred +1)+1;
        end
        CMdisplay = zeros(nclasses+1,nclasses+1);
        CMdisplay(1,2:11)=0:9;
        CMdisplay(2:11,1)=0:9;
        CMdisplay(2:11,2:11)=CM;
        fprintf('CM \n\n')
        disp(CMdisplay);

    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MPL Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%

neurons = [50,100];
if choice == 3
    fprintf('Accuracy Testing for MultiLayer Perception: \n')

    hiddenNeurons = size(neurons,2);
    for a = 1:hiddenNeurons % a is the index of the network of neurons
        val = neurons(a);
        mlpTest = fitcnet(newTrainImages,newTrainLabels,'LayerSizes',val); % Running MLP
        [Ytest,Score]=predict(mlpTest,newTestImages); %Prediction Aspect
        CM = confusionmat(newTestLabels, Ytest);
        
    
        accuracy = sum(Ytest == newTestLabels) / length(newTestLabels); 
        fprintf(' %d neurons: %.4f\n\n', val, accuracy*100); %Accuracy is a probability
        fprintf('CM \n\n')

        CMdisplay = zeros(nclasses+1,nclasses+1);
        CMdisplay(1,2:11)=0:9;
        CMdisplay(2:11,1)=0:9;
        CMdisplay(2:11,2:11)=CM;

        fprintf('CM')
        disp(CMdisplay);
       
    end

end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CNN Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%

if choice == 4
    fprintf('Accuracy Testing for CNN: \n')

    %Have to reshape it a bit differently for this part of the code. So had
    %to copy and paste some of the code
    
    load("data_mnist_test_original.mat") 
    testImages = reshape(imgs,28*28,10000)'; % <--This changed
    testLabels = labels; %No need to reshape the labels. Only the Images
    newTestImages = [];
    newTestLabels = [];
    
    for i = 1:nclasses %Looping through all the classes to obtain the 500... will get 5000 examples in total
        cindex = find(testLabels == (i-1));% We want to get to 0 not 1
        cindex = cindex(1:nexamples);
    
        %Creating new dataset/dataframe
        newTestImages = [newTestImages; testImages(cindex,:)];
        newTestLabels = [newTestLabels; testLabels(cindex)];
    end

    %Normalizing the Data
    newTestImages = double(newTestImages)/255;
    
    %Images
    reshapeTestImage = reshape(newTestImages',28,28,1,[]);
    
    %Categorical of labels
    CatTestLabel = categorical(newTestLabels);
    
    load("data_mnist_train_original.mat") %Loading the training set
    trainImages = reshape(imgs, 28*28,60000)';
    trainLabels = labels; %No need to reshape the labels. Only the Images
    newTrainImages = [];
    newTrainLabels = [];
    
    for i = 1:nclasses %Looping through all the classes to obtain the 500 per class... will get 5000 examples in total
        cindex1 = find(trainLabels == (i-1));% We want to get to 0 not 1 and selecting the first 500
        cindex1 = cindex1(1:nexamples); 
    
        %Creating new dataset/dataframe
        newTrainImages = [newTrainImages; trainImages(cindex1,:)];
        newTrainLabels = [newTrainLabels; trainLabels(cindex1)];
    end
    
    %Normalizing the Data
    newTrainImages = double(newTrainImages)/255;

    %Images
    reshapeTrainImage = reshape(newTrainImages',28,28,1,[]);
    
    %Categorical of labels
    CatTrainLabel = categorical(newTrainLabels);

    % The actually network
    layers = [
        imageInputLayer([28 28 1])

        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];

    % Display the network
    %analyzeNetwork(layers);

    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{reshapeTestImage,CatTestLabel}, ...
        'ValidationFrequency',30, ...
        'Verbose',false, ...
        'Plots','training-progress')

    %Network
    net = trainNetwork(reshapeTrainImage,CatTrainLabel,layers,options);

    %Prediction of Labels
    Yout = classify(net,reshapeTestImage);
    Ytest = categorical(newTestLabels);

    %Accuracy
    accuracy = sum(Yout == Ytest)/numel(Ytest);
    fprintf('Accuracy: %.4f%%\n', accuracy*100); % Accuracy is probabilit

    fprintf('CM: \n');

    CM = confusionmat(Ytest,Yout);
    CMdisplay = zeros(nclasses+1,nclasses+1);
    CMdisplay(1,2:11)=0:9;
    CMdisplay(2:11,1)=0:9;
    CMdisplay(2:11,2:11)=CM;

    disp(CMdisplay);
      

end

