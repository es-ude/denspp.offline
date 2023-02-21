clear
clc
%---------------------------------------------------------------------------
% openNEV('rps_20131016-115158-NSP1-001.nev');    % NEV: Neural EVents, threshold crossed spike waveforms(Neural Spiking Data)
                                                   %'NEV.Data.Spikes.TimeStamp' (time stamps of all spikes)

load('rps_20131016-115158-NSP1-001.mat');         % Neural Spiking Data

%---------------------------------------------------------------------------
behavioralData=load('20131016-115158_rps_01_behav.mat');          % Behavioural Data 

%---------------------------------------------------------------------------
tasksInfo=behavioralData.saveData.EventTimes;   % 'behavioralData.saveData.EventTimes'
                                                % timing information of trails and event in a session
                                                % m:total Number of trials,
                                                % n:total Number of events in one trial

 totalTasks=size(tasksInfo,1);  %total number of trials in one session
            
%---------------------------------------------------------------------------            
%unsortedSpikesSession: is structure array containg 96 fields
% where each field is n*3 matrix
% n is the total number of spiking activity of neuron at particular electrode during one complete session
% 3 cloumn are [indices of occurance of spike, time stamp of nsp, electrode number]

timeStamps=NEV.Data.Spikes.TimeStamp;     %Timestamps of spikes
electrode=NEV.Data.Spikes.Electrode;      %Occurance of spike on specefic electrode
electrodeUsed = unique(NEV.Data.Spikes.Electrode); %list of electrode used
totalElectrodeUsed=length(electrodeUsed);

for k=1:1:totalElectrodeUsed                %iterate through each electrode (e.g 1 to 96)
    IndicesElectrode=find(electrode==k);        %indices of a particular electrode
    electrodeID=electrode(IndicesElectrode);    %Electrode ID
    timeStampsElectrode=timeStamps(IndicesElectrode);%getting the time stamps of spikes
    unsortedSpikesSession.data{1,k}=[IndicesElectrode;timeStampsElectrode;electrodeID]';
    %just for the sake of my convience, transposing the matrix and assigining to one of struct
end

%---------------------------------------------------------------------------            
% extract feature, Unsorted
                                                        
% startingIndexOffset=[2,0.4];          %[startingIndex(of event), startingOffset(in seconds)]
% endingIndexOffset=[7,0.6];            %[endingIndex(of event), endingOffset(in seconds)]

startingIndex=2;    %startingIndex(of event)
startingOffset=0.4; %startingOffset(in seconds)
endingIndex=7;      %endingIndex(of event)
endingOffset=0.6;   %endingOffset(in seconds)

featuresVectorsUnsorted = zeros(totalTasks,totalElectrodeUsed); %Preallaoction of memory


for i=1:totalTasks  %iterate through each trial and will extract features (Total Trials/Tasks=50)

    startingTime_i=tasksInfo(i,startingIndex)*30000 +startingOffset*30000; %30000 is smapling frequency
    %mapping time(seconds) into NSP sampling time
    
    endingTime_i=tasksInfo(i,endingIndex)*30000 +endingOffset*30000; %30000 is sampling frequency
    %mapping time(seconds) into NSP sampling time

    featuresVector_i=zeros(totalElectrodeUsed,1); %(e.g 96*1)
    %pre-allocation for one feature vector, coressponding to each trial
    
    for j=1:totalElectrodeUsed %for every trial, iterrate through each and every electrode to come up with one
        %feature vector

        spikingActivityElectrode_j = find(unsortedSpikesSession.data{1, j}(:,2)>=startingTime_i & unsortedSpikesSession.data{1, j}(:,2)<=endingTime_i);
        % calculating  for each electrode, total number of spikes between startingTime_i and endingTime_i of each trial
        
        feature_j= numel(spikingActivityElectrode_j);%just adding the spikes
        featuresVector_i(j)=feature_j;%each iteration gives me one feature for one task
    end

    featuresVectorsUnsorted(i,:)=featuresVector_i';
    %come up with one feature vector after one time i loop and 96 times j loop

end
%---------------------------------------------------------------------------

% Extract Labels(for each task in one complete session including unsuccesful trials)

labelsInt=zeros(1,totalTasks);      %Preallocation, and labelsInt will contain labels of all task as integer
chckOutliers=zeros(1,totalTasks);   %preallocation, chckOutlier=1 if the trial is succesful and =0 if the trial is unsuccesful

for jj=1:1:totalTasks %itterate thoroug each and evaery trial of the session and check weather the trial was succesful or not

    buttonPressed=behavioralData.saveData.Trials(jj).ButtonPressed; %feedback from the subject

    actionType=behavioralData.saveData.Trials(jj).ActionType;    %asked to do
    labels = behavioralData.saveData.Trials(jj).ActionType;      %label of the 'asked to do' in the form of string
    strLabel=labels(1,:);
    %checkOutlier_jj=checkOutlier(buttonPressed_jj,actionType_jj);% This static function will detect the weather the
    % task was succesful or not, return 1 if it is succesful, 0
    findSpacesButtonPressed=isspace(buttonPressed); %before comparing check weather there is a presence space
    findSpacesActionType=isspace(actionType);% before comapring checking for space

    findSpacesButtonPressed=find(findSpacesButtonPressed==1); %if space is present, will get the index of space
    findSpacesActionType=find(findSpacesActionType==1);% if space is present, will get the index of it
    if ~isempty(findSpacesButtonPressed) %if space is present,get rid of space
        buttonPressed=buttonPressed(1:findSpacesButtonPressed-1);
    end
    if ~isempty(findSpacesActionType)% if space is present, get rid of it
        actionType=actionType(1:findSpacesActionType-1);
    end
    % comaparing asked action and performed action
    status=strcmp(buttonPressed,actionType);
    % if both are same function will return 1 otherwise it will return 0
    % otherwise


    %labelsInt(jj)=GetLabels.convertLabelStrInt(labelStr);% This static function will convert string label into integer label
    findSpaces=isspace(strLabel); %Given labels may have space

    findSpaces=find(findSpaces==1); %find if there is any space

    if ~isempty(findSpaces)         %if there is space
        strLabel=strLabel(1:findSpaces-1);%will extract all the alphabet and discard the space if any
    end
    if strcmp('rock',strLabel)==1 %Now compare the string of label with one of posssible outcome
        strLabel=1;             % if the srLabel is same as 'rock' then it will be assigned as an integer label
        %'1'
    elseif strcmp('paper',strLabel)==1% second case is comparing with another possible out come called 'paper'
        strLabel=2;                     % if it lies in the same category, it will be assigned as label '2'
    elseif strcmp('scissors',strLabel)==1% third case is another possible outcome, so comparing input parameter
        % with this possibilty, if
        % it les inthis category it
        % will be assigned as '3'
        strLabel=3;
    else
        strLabel=4;  %This will never happen usless there is a bug in the code
    end
    
    
    intLabel=strLabel; % will assign the feature to intLabel, which is an output argument
   % chckOutliers(jj)=checkOutlier;
end
%---------------------------------------------------------------------------
%Extract feature vectors and labels for successful trials
% 0 in the case when the trial was unsuccessful and 1 when the trial was considered to be successful

indicesRequired=find(chckOutliers==1);% will give the index of successful trials
extractedLabels=labelsInt(indicesRequired);%will give the labels of succesful trials
extractedFeaturesVector=(featuresVectorsUnsorted(indicesRequired,:))';% will giv the feature vectors of succesfull trails

%---------------------------------------------------------------------------
% %Splitting the data into Training and Test Data
% 
% %varargin: is optinal paramter and expect scalar real value
% %forexample: if you want to slice your data with 80%
% %training Data and 20% percent test data. then
% %varargin{1}=0.8
% %if optional parameter is not provided then by default data
% %will be ditributed automaticall in 80% training and 20% test
% 
% x=featuresVectorsUnsorted; % matrix of all features vectors
% t=labels; %labels
% Q = size(x,2); %total number of inputs
% if length(varargin)>=1
%     Q1 = floor(Q*varargin{1});
%     Q2 = Q-Q1;
% else
%     Q1 = floor(Q*0.80); % total number of inputs selected for training
%     Q2 = Q-Q1;  %total_number of inputs seleccted for test purpose
% 
% end
% ind = randperm(Q);%producing 1:Q, random numbers
% ind1 = ind(1:Q1); % first 80% is slected for training
% ind2=ind(Q1+(1:Q2));% rest of 20% is selected for test purpose
% 
% trainingData.data = x(:,ind1); %train_x
% trainingData.labels = t(:,ind1);% train_y
% testData.data = x(:,ind2); %test_x
% testData.labels = t(:,ind2);% test_y