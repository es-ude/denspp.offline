classdef GetLabels
    %GetLabels:
    
    %Explantion:Objective of this class is to extract labels associated with
    %trial, some of the trials are successful and some of the trial are
    %unsuccesful. So, it worthy to identify those trials which are
    %successful and remaining are unsuccessful.This class will extract
    %labels and identify successful trials
    
    %Object: object of this class is 'self'
    
    %Properties:To extract labels only behavioral data is required. So,
    %behavioral data is property of this class
    
    %behavioralData:is of datatype struct and it contain all the necessary
    %inoformation that describes the trial and session. This struct has
    %field 'SaveData', which consist of multiple structs. However, the
    %struct in which we are intertsed in is 'Trials'.
    
    properties
        % Acess attrbute of properties is remained private beacause
        % Data class which inherits from FeaturesExtraction and GetLabels
        %both requires behavioral data
        behavioralData
    end
    methods
        function self=GetLabels(behavioralData)
            % constructor requires the beahvaioral data and this function
            % expect it to be in the form of struct
            self.behavioralData=behavioralData;
        end
        
        function [labelsInt,chckOutliers]=extractLabelsSession(self)
            % Explanation:This function will extract the labels for each task in one
            % complete session including unsuccesful trials
            
            %inputParameters: it will take object 'self' of this class as an input
            %argument
            totalTasks=length(self.behavioralData.saveData.EventTimes); % total amout task in one session
            labelsInt=zeros(1,totalTasks); %Preallocation, and labelsInt will contain labels of all task as integer
            chckOutliers=zeros(1,totalTasks);%preallocation, chckOutlier will be one if the trial is succesful and zero if the
            % trial is unsuccesful
            for jj=1:1:totalTasks %this loop will itterate thoroug each and evaery trial of the session and check weather the trial was succesful
                %or not
                buttonPressed_jj=self.behavioralData.saveData.Trials(jj).ButtonPressed; %feedback from the subject
                actionType_jj=self.behavioralData.saveData.Trials(jj).ActionType;%asked to do
                labels_jj = self.behavioralData.saveData.Trials(jj).ActionType; %label of the 'asked to do' in the form of string
                labelStr=labels_jj(1,:);
                
                checkOutlier_jj=GetLabels.checkOutlier(buttonPressed_jj,actionType_jj);% This static functiion will detect the weather the
                % task was succesful or not, return 1 if it is succesful, 0
                % otherwise
                labelsInt(jj)=GetLabels.convertLabelStrInt(labelStr);% This static function will convert string label into integer label
                chckOutliers(jj)=checkOutlier_jj;
            end
            
        end
    end
    
    
    
    methods (Static,Access=private)
        
        function intLabel= convertLabelStrInt(strLabel)
            % Explanation:This function will convert the chararcter
            % array(string) into integer
            %inputParemters:
            %1)strLabel: Label of one trial as string in the session
            %outputParameters:
            %intLabel: This is the corresponding label in integer
            
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
            
        end
        function status=checkOutlier(buttonPressed,actionType)
            %Explanation: This function will check if the trial is
            %succesfully executed or mistake has been made by subject 
            
            %inputParamter:
            %1)buttonPrssed: Type: string
            %subject intension
            %2)actionType: Type:string
            %actuall task
            
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
        end
    end
end


