clear all
close all
clc
warning off

[ff,pp] = uigetfile('*.*');

newImage = imread([pp ff]);
im = newImage;
% ====================================================================
% = CNN
disp('=================================')
disp(' CNN Algorithm')
disp('=================================')

imds = imageDatastore('Dataset', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imshow(imds.Files{1});
labelCount = countEachLabel(imds);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize');
inputSize=[450 600 3];
imdsTrain=augmentedImageDatastore(inputSize, imdsTrain);
imdsValidation=augmentedImageDatastore(inputSize, imdsValidation);

layers = [
    imageInputLayer([450 600 3])
    
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
    
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress');
nets = trainNetwork(imdsTrain,layers,options);
net1 = trainNetwork(imdsTrain,layers,options);


nets.Layers

inputSize = nets.Layers(1).InputSize;
analyzeNetwork(nets)


layersTransfer = nets.Layers(1:end-3);


categories = {'merkel cells tissue','Melanoma'};
imds = imageDatastore('Dataset', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

tampered = find(imds.Labels == 'merkel cells tissue', 1);
scale = find(imds.Labels == 'Melanoma', 1);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize');

numClasses = numel(categories(imdsTrain.Labels));


imageSize = nets.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize,imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize,imdsValidation, 'ColorPreprocessing', 'gray2rgb');
featureLayer = 'fc';
trainingFeatures = activations(nets, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
trainingLabels = imdsTrain.Labels;


classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
testFeatures = activations(nets, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
testLabels = imdsValidation.Labels;
testImage = readimage(imdsValidation,1);
testLabel = imdsValidation.Labels(1);


ds = augmentedImageDatastore(imageSize,newImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(nets, ds , featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
label = predict(classifier,imageFeatures , 'ObservationsIn', 'columns');
sprintf('the image is %s class', label)


figure,
imshow(newImage);
title(['The image class (CNN) = ', label])

% Evaluate the performance of the network
predictedLabels_class = classify(net1, imdsTrain);

accuracy_cnn = mean(predictedLabels_class == imdsTrain.Labels);
disp(['CNN accuracy: ', num2str(accuracy_cnn*100),' %']);

% ====================================================================

% ================
% === DNN ========
disp('=================================')
disp(' DNN Algorithm')
disp('=================================')

% Load the digit dataset
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'digitDataset');
imds = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomize');
labelCount = countEachLabel(imds);

% Create a DNN architecture
layers = [
    imageInputLayer([450 600 3])
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress');

% Train the DNN
net = trainNetwork(imdsTrain, layers, options);

% Evaluate the DNN on the test set
YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

% Calculate the accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test accuracy DNN: %.2f%%\n', accuracy * 100);

categories = {'merkel cells tissue','Melanoma'};
tampered = find(imds.Labels == 'Actinic keratosis', 1);
scale = find(imds.Labels == 'Melanoma', 1);

numClasses = numel(categories(imdsTrain.Labels));

imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize,imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize,imdsValidation, 'ColorPreprocessing', 'gray2rgb');
featureLayer = 'relu';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
trainingLabels = imdsTrain.Labels;


classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
testLabels = imdsValidation.Labels;
testImage = readimage(imdsValidation,1);
testLabel = imdsValidation.Labels(1);
ds = augmentedImageDatastore(imageSize,newImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, ds , featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
label = predict(classifier,imageFeatures , 'ObservationsIn', 'columns');
sprintf('the image is %s class', label)


figure,
imshow(newImage);
title(['DNN The image class = ', label])


% Evaluate the performance of the network
predictedLabels_class = classify(net, imdsTrain);

accuracy_dnn = mean(predictedLabels_class == trainingLabels);
disp('=================================')
disp(' ')
disp(['DNN accuracy: ', num2str(accuracy_dnn*100),' %']);
disp('=================================')

% ====================================================================
% ==== RNN ==============
disp('=================================')
disp(' RNN Algorithm')
disp('=================================')

% Load and preprocess an image

% Load and preprocess an image
im = im2double(im);

% Flatten the image into a 1D sequence
imSeq = reshape(im, [], 1);

layers = [
    imageInputLayer([450 600 3])
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
    
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Specify training options
options1 = trainingOptions('sgdm', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'Plots', 'training-progress');
% Create a datastore for the image

imds = imageDatastore('Dataset', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomize');

% Create a sequence datastore from the image
seqData = transform(imdsTrain, @(data) {imSeq});

% Train the LSTM
net = trainNetwork(imdsTrain, layers, options);

% Classify the image
YPred = classify(net, im);

% Display the classification result
fprintf('Image is classified as: %s\n', YPred);

figure,
imshow(newImage);
title(['RNN The image class = ', YPred])

% Evaluate the performance of the network
predictedLabels_class = classify(net, imdsTrain);

accuracy_rnn = mean(predictedLabels_class == imdsTrain.Labels);
disp('=================================')
disp(['RNN accuracy: ', num2str(accuracy_rnn*100),' %']);
disp('=================================')
disp(' ')
disp(' ')
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.5,'randomize');

% ======================================
disp('=================================')
disp(' RESNET50 Algorithm')
disp('=================================')

% RESNET50
% Create a simple ResNet-50 architecture
layers = [
    imageInputLayer([224 224 3], 'Name', 'input')
    
%     convolution2dLayer(7, 64, 'Stride', 2, 'Padding', 3, 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn_conv1')
    reluLayer('Name', 'relu_conv1')
    maxPooling2dLayer(3, 'Stride', 2, 'Padding', 1, 'Name', 'pool1')
    
    % Convolutional blocks
    createResNetBlock(64, [3, 64], '1', 'a', 1)
    createResNetBlock(256, [3, 64], '2', 'a', 2)
    createResNetBlock(512, [3, 64], '3', 'a', 2)
    
    % Add fully connected and classification layers
    fullyConnectedLayer(1000, 'Name', 'fc_1000')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];


% Create a layer graph and view it
lgraph = layerGraph(layers);
figure,plot(lgraph);

% Specify training options and train the network
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', 64, ...
    'Verbose', true);

% Load your dataset and train the network
% Make sure to specify your own data and labels

% Train the network
YTest = imdsTrain.Labels;

% net = trainNetwork(imdsTrain, lgraph, options);
% Make predictions using the trained network
predictedLabels = classify(net, im);

figure,imshow(im);
title(['Resnet50',predictedLabels])
disp(['Resnet50',predictedLabels])

% Evaluate the performance of the network
predictedLabels_class = classify(net, imdsTrain);

accuracy = mean(predictedLabels_class == YTest);
disp('=================================')
disp(' ')
disp(['Resnet-50 Classification accuracy: ', num2str(accuracy*100),' %']);
disp('=================================')

%RNN accuracy: 78.5714 %
%DNN accuracy: 39.2857 %
%CNN accuracy: 87.5 %
%Resnet-50 Classification accuracy: 85 %
