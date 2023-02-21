clear
clc

% Data
%bd='C:\Users\KlaesLab03\Desktop\OneDrive - rub.de\MatlabScripts\PhDStuff\BCIPackages\Data\20131016-115158_rps_01_behav.mat';
bd='20131016-115158_rps_01_behav.mat'; %Behavioural Data

%nsd='C:\Users\KlaesLab03\Desktop\OneDrive - rub.de\MatlabScripts\PhDStuff\BCIPackages\Data\rps_20131016-115158-NSP1-001.mat';
nsd='rps_20131016-115158-NSP1-001.mat'; %Neural Spiking Data
startingIndexOffset=[2,0.4];
endingIndexOffset=[7,0.6];

%1st class(Extract features vectors and labels for different classifiers and regression algo such as Neural Network)
data=Data(startingIndexOffset,endingIndexOffset,bd,nsd); %inititaing constructor
%[featVect,labels]=featuresVectorsAndLabelsUnsorted(data);
 
%[extractedFeaturesUnsoretedVector,extractedUnsortedLabels]=data.featuresVectorsAndLabelsUnsorted();
[trainingData,testData]=data.featuresVectorsAndLabelsUnsorted();

%[extractedFeaturesSortedVector,extractedSortedLabels]=data.featuresVectorsAndLabelsSorted();


%cmpFeatLab=CompatibleFeaturesLabels(extractedFeaturesUnsoretedVector,extractedUnsortedLabels');
%[featNN,labNN]=cmpFeatLab.makeComp4NN();
%[featNN,labNN]=Decoder.NeuralNetworks.makeComp4NN(extractedFeaturesUnsoretedVector,extractedUnsortedLabels');

%[trainingData,testData]=Data.getTrainingTestdata(featNN,labNN);

% [trainingData,testData]=GetTrainingTestData.dataLoader4NN(featNN,labNN);


layers=[30,15];

nnDec=Decoder.NeuralNetworks(layers);
nnDec.train(trainingData,testData);

nnDec.test(testData.data(:,1))



%similarly for LFP features, i have to write a fucntion that will extract
%LFP features for whole session


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% decoder=Decoders(extractedFeaturesSortedVector,extractedSortedLabels);
% 
% dataDistRatio=0.8;
% layers=[30,15];
% 
% [featNN,labNN]=decoder.neuralNetworks(dataDistRatio,layers);



















































%2nd Class
%  compfeatlab=CompatibleFeaturesLabels(extracted0--------------SortedVector,extractedSortedLabels);
%  [featNN,labNN]=compfeatlab.makeComp4NN();
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %Data distribution
%  trainingRatio=0.8;
%  [trainingData,testData]=DataDistributionRatio.dataLoader4NN(featNN,labNN);
% 
%  layers=[30,15];
%  nnModels=Decoders(layers,trainingData,testData);
% [netsTrained,performance]=nnModels.neuralNetworks();





















 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
%  b=GetLabels(bd);
% [labels_int,chck_outliers]=extractLabelsSession(b);









