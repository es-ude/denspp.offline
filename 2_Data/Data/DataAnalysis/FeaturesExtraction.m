classdef FeaturesExtraction
    
    %FeaturesExtraction:
    
    %Explanation:
    
    %Objective: Objective of this class is to extract all kind of
    %features vectors(sortedSPikes,unsortedSPikes,LocalFieldPotential)
    %between two specefic events of trials
    
    %Class Methods:
    
    %extractUnsortedFeatures:
    %The function of this class method is to extract unsorted spikes
    %Feature vectors for one complete session detail about input parameters
    %and output paramters can be seen in the function defination
    
    %extractSortedFeatures:
    %The function of this class method is to extract sorted spikes
    %Feature vectors for one complete session detail about input parameters
    %and output paramters can be seen in the function defination
    
    %extractLFPfeatures:
    %Coming soon:)
    
    %Static Methods:
    %Details can be found in the defination of each function
    
    %Object: object of this class is self
    
    %properties:
    
    %1)StartingIndexOffset: this is vector of dimension 1*2
    %first element of vector would be strating index and second index
    %would be starting offset, [startingInd, startingOffset]
    
    %2)EndingIndexOffset: this is the vector of size 1*2
    %first element of vetor would be ending index and second element of
    %list would be ending offset,[endingInd, endingOffset]
    
    %3)behavioralData: is a struct conating all the information regarding
    %trials in different fields. Here'behavioralData.saveData.EventTimes'
    % is field and is used to extract features. This field contains timing
    % information of trails and within trials(event). It is
    % is matrix of size m*n
    % m: total Number of trials
    % n: total Number of events in one trials
    
    %4)NeuralSpikesData:This is a struct and it contain the neural information
    %of trials for one complete session and it has two main fields which
    %are structs
    
    %NEV: This field is a struct and it will contain all the necessary
    %inforamtion to extract unsorted spikes features vectors in its
    %different fields which are of different data types forexample:
    %'neuralSpikesData.NEV.Data.Spikes.TimeStamp' is field of 'NEV'and it is
    %vector of size n*1, where n repesents the time stamps of all spikes in
    % one session. To Extract unsorted spikes features 'nev' struct is used
    
    %sortedSPikes: is a struct and it is in conjuction with 'NEV'struct is
    %used to to extract sorted features vectors. this struct has different one 
    %fields and this filed consis of (n*1) cells where n repesent the number
    %of implanted electrodes(sensors) and each cell contain the sorted spikes
    %information of each electrode.
    
  
    properties
        startingIndexOffset
        endingIndexOffset 
        behavioralData  
        neuralSpikesData 
    end
    
    methods
        function self =FeaturesExtraction(startingIndexOffset,endingIndexOffset,behavioralData,neuralSpikesData)
            %constructor:
            
            self.startingIndexOffset=startingIndexOffset;
            self.endingIndexOffset=endingIndexOffset;
            self.behavioralData=behavioralData;
            self.neuralSpikesData=neuralSpikesData;
        end
        
        function featuresVectorsUnsorted =extractUnsortedFeatures(self)
            %Explanation:
            % This function will be used to extract Unsorted features vectors
            % for one complete session between two events of trials
            
            %InputParameters:
            %Self: is an object of this class for explantion of each
            %property see at the top of script just after the defiantion of
            %class
            
            %output parameters:
            %FeaturesExtractionUnsorted: is m*n matrix, where m is total
            %number of trials and n is total number of fetaures in one
            
            
            tasksInformationSession=self.behavioralData.saveData.EventTimes; %task timings
            %complete timing information of all the trials in one session
            %it should be matrix of size (m*n)
            %m:total number of trials
            %n:total number of events in one trial 
            
            totalTasks=size(tasksInformationSession,1);%total trials in one session
            
            
            unsortedSpikesSession =unsortedDataSession(self);% calling 
            % unsortedDataSession function for explantion see at the
            % defination of function
            totalElectrodeUsed=length(unsortedSpikesSession.data); %Total number of eletcrode that record data
            featuresVectorsUnsorted = zeros(totalTasks,totalElectrodeUsed); %Preallaoction of memory 
            for ii=1:totalTasks  %iterate throuugh each trial and will extract features 
                
                startingIndex=self.startingIndexOffset(1); %starting index
                startingOffset=self.startingIndexOffset(2); %straing offset
                startingTime_ii=tasksInformationSession(ii,startingIndex)*30000 +startingOffset*30000; %30000 is smapling frequency
                %mapping time(seconds) into NSP sampling time  
                endingIndex=self.endingIndexOffset(1); %Ending Index
                endingOffset=self.endingIndexOffset(2);%Ending offset
                endingTime_ii=tasksInformationSession(ii,endingIndex)*30000 +endingOffset*30000; %30000 is sampling frequency
                %mapping time(seconds) into NSP sampling time 
                featuresVector_ii=zeros(totalElectrodeUsed,1);
                %pre-allocation for one feature vector, coressponding to
                %each trial
                for jj=1:totalElectrodeUsed %for every trial, iterrate through each and every electrode to come up with one 
                                            %features vectors
                    spikingActivityElectrode_jj = find(unsortedSpikesSession.data{1, jj}(:,2)>=startingTime_ii & unsortedSpikesSession.data{1, jj}(:,2)<=endingTime_ii);
                    % I am calculating  for each electrode, total number of
                    % spikes between starting_ii and ending_ii for each trial 
                    feature_jj= numel(spikingActivityElectrode_jj);%just adding the spikes 
                    featuresVector_ii(jj)=feature_jj;%each iteration gives me one feature for one task
                end
                
                featuresVectorsUnsorted(ii,:)=featuresVector_ii';%come up with one feature vector after one time ii loop and
                %96 time jj loop
                
            end    
        end
        
        
        
        
        function unsortedSpikesSession =unsortedDataSession(self)
            %Explantion: This function will take the struct
            %self.neuralSpikesData.nev as an input argument,
            %and will construct 96 matrix out of it, each matrix
            %corresponds to one electrode. and each matrix have dimension
            %n*3. And each matrix is assign to one struct
            
            %input Parameter: following properties of class will be used 
            %self.neuralSpikesData: contains the struct of .NEV of coresponding
            %neural activity
            
            %outputParamter: is of type strtuct
            %unsortedSpikesSession:
            %is struct contain 96 fields
            % where each field is n*3 matrix
            % n is total number of spiking activity of neuron at particular 
            % electrode during one complete session%3 clumns are the 
            %index of occurance of spike,time stamp of nsp and electrode 
            %number
            
            
            timeStampsSpikes=self.neuralSpikesData.NEV.Data.Spikes.TimeStamp; %Timestamps of spikes
            electrodeSpikes=self.neuralSpikesData.NEV.Data.Spikes.Electrode;   %Occurance of spike on specefic electrode
            electrodeUsed = unique(self.neuralSpikesData.NEV.Data.Spikes.Electrode); %list of electrode used
            for kk=1:1:length(electrodeUsed) %iterate through each electtrod
                spikesIndicesElectrode_kk=find(electrodeSpikes==kk); %indices of spikes at particular electrode
                electrodeId_kk=electrodeSpikes(spikesIndicesElectrode_kk);%getting identity of each spike
                timeStampsElectrode_kk=timeStampsSpikes(spikesIndicesElectrode_kk);%getting the time stamps of spikes
                unsortedSpikesSession.data{1,kk}=[spikesIndicesElectrode_kk;timeStampsElectrode_kk;electrodeId_kk]';
                %just for the sake of my convience, transposing the matrix
                %and assigining to one of struct 
            end
            
        end
        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function featuresVetctorsSession=extractSortedFeatures(self)
            %Explanation: This function will extract sorted features vectors
            % for one complete session between two events of trial 
            % and this will use NEV struct in conjction with sortedSpikes strucut 
            % which are fields of Self.NeuralSpikesData
            
            %self is object of this class and is well above
            
            %featuresVectorSorted
            %is m*n matrix, where m is total
            %number of trials and n is total number of fetaures in one
            %trial
            % calling sortedDataSession, details: see defination of function
            
            sortedDataElectrodeSession = sortedDataSession(self);
            % sortedDataElectrodeSeesion: will be a struct and it will be
            % having 96 cells as one field. each cell which will be a matrix
            %coresponds to one electrode and this matrix will be of
            %dimension n*3, n is total number of spikes at this particular
            %elctrode in one session and  column would repesent the
            %following information
            %1)spike is coming from which unit
            %2) index of spike
            %3)time stampp of particular spike
            
            [semiFinalSortedUnitsLabels,accumulativeActivityUnitsResult]=semiFinalFeaturesSorted(sortedDataElectrodeSession,self);
             %accumulativeActivityUnitsResult: it is of datatype struct
            %and it will contain one field, and this field will then
            %contain 'n' number of structs, where 'n' is total number of
            %trials in one session. Each struct in field contain '1*m' number
            %of cells, where 'm' is total number of implanted electrode.
            %Each cell then contain a list '1*p', where p is total number
            %of cells at particular electrode and each element of this list
            %is integer nad it contain total number of firing activity
            %between to event with offsets, which are well defiined. and
            %each element of '1*p' vector will later on consiered as one
            %feature.
            electrodeImplanted=length(accumulativeActivityUnitsResult.data{1, 1}.sortedSpikesUnit);%
            totalTasksSession=length(accumulativeActivityUnitsResult.data);
            % calculating the index of electrode which contain spikes other than
            % artifacts
            index=1;
            totalNumberUnits=0;
            %calculating the size of features vaector
            for ii=1:electrodeImplanted %for each electrode
                if ~isempty(accumulativeActivityUnitsResult.data{1, 1}.sortedSpikesUnit{1, ii}) %if a particular eclectrode is[]
                                                                                                 %do not include it
                    numberUnits=length(accumulativeActivityUnitsResult.data{1, 1}.sortedSpikesUnit{1, ii});%number of units found on each electrode
                    indexMeaningfulData(index)=ii;%calculating the index of electrodes, having meaningful data
                    index=index+1; %now increasing the index by 1
                    totalNumberUnits=totalNumberUnits+numberUnits;%add feature or features in the list of features vector 
                end
            end
            
            data=zeros(1,totalNumberUnits); %Now, as we know the size of features vecto, we can preallocate the psace in meamory
            featuresVetctorsSession=zeros(totalTasksSession,totalNumberUnits);% pre-allocation for all the features vectors in the session 
            for jj=1:totalTasksSession %for each trial extarct features vector
                startingIndex=1;
                for kk=1:length(indexMeaningfulData) %for each session, for each electrode having units 
                    newData=accumulativeActivityUnitsResult.data{1, jj}.sortedSpikesUnit{1,indexMeaningfulData(kk)};%data at each electrode
                    unitsElectrode=length(newData);
                    endingIndex=startingIndex;
                    if kk==1 && unitsElectrode >1
                        endingIndex=endingIndex+unitsElectrode;
                        
                    end
                    if kk>1 && unitsElectrode >1
                        endingIndex=endingIndex+unitsElectrode-1;
                    end
                    data(startingIndex:endingIndex)=newData;%constructing features vector by adding features vector every time
                    startingIndex=startingIndex+unitsElectrode;%straing eindex of next feature
                    
                end
                %feature_vectors_final.data{jj}=data;
                featuresVetctorsSession(jj,:)=data; %addding one feature at a time
                data=[];
            end
            
        end
        function sortedSpikesSession = sortedDataSession(self)
            %inputParameters:
            % This function takes self.NeuralSpikeData
            % and it  conatin two struct(NEV,sortedspikes)
            % 1)NEV is a struct conatin the unsorted spiking activity of cells(neuron) on all
            % electrodes(channels)
            % 2) SortedSpikes is  a struct contains the information about sorted spikes
            % of each channels, each channel consist of multiple units.
            % The SPikes are sorted using Gausian Mixture Model(GMM)
            % This function process the raw information comming from .NEV file and
            %sorted_spikes;
           
            %-----------------------------------------------------------------------------
            % First extract all the necessary information from nev construct
            %1) indices of spikes
            %2) time stamp of spikes
            %3) Data of each spike(not needed to process further)
            %----------------------------------------------------------------------
            % Spikes ocuurance: 
            spikesOccuranceAccumulative = self.neuralSpikesData.NEV.Data.Spikes.Electrode;
            %time stamp of each spike:
            spikesTimeStampAccumulative = self.neuralSpikesData.NEV.Data.Spikes.TimeStamp;
            %struct:contains sorted spikes per electrode for complete one task
            sortedSpikesRaw = self.neuralSpikesData.sortedSpikes;
            % total number of used electrode
            electrodeUsed = unique(self.neuralSpikesData.NEV.Data.Spikes.Electrode);
            %just another check to avoid any kind bug in the code, not mandatory
            if length(spikesOccuranceAccumulative)==length(spikesTimeStampAccumulative)
                for ii=1:1:length(electrodeUsed) %iterate through each elaectrode 
                    % find the all spikes of every individual electrode
                    spikesIndicesIndividualElectrode = find(spikesOccuranceAccumulative==ii);
                    %time stamp of spikes at each electrode for one
                    %complete session
                    spikesTimeStampsIndividualElectrode =spikesTimeStampAccumulative(spikesIndicesIndividualElectrode);
                    %which spike is comming from which neuron(cell) at each
                    %electrode
                    unitsActivityPerElectrode=sortedSpikesRaw{1, ii}.index;
                    %just another check for bug, not mandatory
                    if length(unitsActivityPerElectrode) ==length(spikesTimeStampsIndividualElectrode)
                        %putting : spikes per unit, corresponding index
                        %spike and its time stamp 
                        information = vertcat(unitsActivityPerElectrode,spikesIndicesIndividualElectrode,spikesTimeStampsIndividualElectrode);
                        %Assiging information to one cell of struct, there
                        %will be 96 cell for struct and for every iteration
                        % i will compe up inforamtion corespond to eaach
                        % electrode
                        sortedSpikesFiltered.information{ii} = information'; 
                    end
                end
            end
            sortedSpikesSession=sortedSpikesFiltered; %return the spiking sorted spike per electrode of compltete one session
            
            
        end
        function [semiFinalSortedUnitsLabels,accumulativeActivityUnitsResult] = semiFinalFeaturesSorted(sortedDataSession,self)
            %Explanation:This function will take 'sortedDataSession', which
            %will assign label to each spike with time stamp and index of 
            %occurance. And the object of this class as an input
            %argument and it will give me two struct as an
            %output argument.these struct , then contain number of 
            %fields equal to number trial in one session. lets say i have 50
            %trials in one session then accumulativeActivityUnitsResult which is 
            %an output argument is of type struct contain 50
            %fields and each field is then struct and this struct contain
            %cells equals to number of implanted electrode. Each field contain 
            %spiking activity of each unit between two event if trials
            %which will then be treated as features at each elctrode and accumulative 
            %affect all electrode will be considerd as one feature vector
            
            %input paramters:
            %sortedDataSession: this input parameter is a struct and is well 
            %explained above
            %self: is an object of this class
            %in this method only self.startingIndexOffset and
            %self.endingIndexOffset will be used to extract features
            %between two events of trials
            
            %output paramters: both output paramters conatin redundatnt
            %information
            %accumulativeActivityUnitsResult: it is of datatype struct
            %and it will contain one field, and this field will then
            %contain 'n' number of structs, where 'n' is total number of
            %trials in one session. Each struct in field contain '1*m' number
            %of cells, where 'm' is total number of implanted electrode.
            %Each cell then contain a list '1*p', where p is total number
            %of cells at particular electrode and each element of this list
            %is integer nad it contain total number of firing activity
            %between to event with offsets, which are well defiined. and
            %each element of '1*p' vector will later on consiered as one
            %feature.
            %'semiFinalSortedUnitsLabels': this is not required later
            %because it conatin some what redundant inofrmation as
            %explained in 'accumulativeActivityUnitsResult'
            %'semiFinalSortedUnitsLabels' contain two fields
            %'sortedSpikesUnit'
            %'sortedSpikesLabels'
            %
            %'sortedSpikesUnit'. conatin exactly same information
            %as 'accumulativeActivityUnitsResult'
            %'sortedSpikesLabels': this contain additional inforamtion, it
            %will conatin 96 cells each cell contain the identity of
            %'cells' forExample: [1,2,255], this list tells me I have
            %1,2,and  255 cells found at one particular electrode
            
            
            tasksWithEvents=self.behavioralData.saveData.EventTimes; %%
            %matrix of dimension: m*n, m=total number of trials and is
            %total number of events in trial, and it will conatin timing
            %inofrmation and which are in seconds
            totalNumberTask = length(tasksWithEvents(:,1)); %total number task in one session
            %total_number_events = length(tasks_with_events(1,:));
            numberChannels=length(sortedDataSession.information); %total number channels
            startingIndex=self.startingIndexOffset(1);%starting index: which will be used to extract feature
            startingOffset=self.startingIndexOffset(2);%straing offset: in conjunction with starting index gives me 
                                                        %exact time, when to start the process of feature extraction
            endingIndex=self.endingIndexOffset(1);%gives me the index of ending event 
            endingOffset=self.endingIndexOffset(2);%tells me when to stop.
            for ii = 1:1:totalNumberTask %%start the process of features extraction starting from trial one to last trial
                eventsTask=tasksWithEvents(ii,:);%will slice one complete row at a time of taskWithEvents, which contain 
                                                 % the complete timing information of one trial including all events  
                spikingActivityUnitsEachTask = FeaturesExtraction.sortedUnitsPerTask(sortedDataSession,eventsTask,numberChannels,startingIndex,startingOffset,endingIndex,endingOffset);
                %calling stattic method of this class 'sortedUnitsPerTask'
                %this function as output argument gives me 'spikingActivityUnitsEachTask', which is of data type struct
                %this struct contain two more struct as fields and there
                %names are 
                % sortedSpikesUnit
                % sorttedUnitsIds
                
                %sortedSpikesUnit: contain 1*m cells, where m is total
                %number of implanted electrodes and every cells is 1*n list
                %and it each entry tells me spiking activity of each unit for one trial 
                %during the the specefic events
                %sorttedUnitsIds: contain 1*m cells, where m is total
                %number of implanted electrodes and every cells is 1*n list
                %and it each entry tells me id of each unit.
                
                for ll=1:numberChannels %for each and every electrode for each trial
                    sortedUnitsS=spikingActivityUnitsEachTask.sortedSpikesUnit{1, ll};%spiking activity of units at each electrode
                    sortedUnitsLabels=spikingActivityUnitsEachTask.sortedUnitsIds{1, ll};%ids of units at each electrode
                    filterNoiseData=sortedUnitsLabels(end);%255: is considerd as artifact so for the removal of spikes from 
                                                            %arifacts
                    if filterNoiseData==255 % getting rid of noise(artifacts)
                        sortedUnitsLabels=sortedUnitsLabels(1:end-1,:); %if last entry of list is 255, don't consider it
                        sortedUnitsS=sortedUnitsS(:,1:end-1);% only consider the spiking of cells having id other than 255
                    end
                    semiFinalSortedUnitsLabels.data{1, ii}.sortedSpikesUnit{1,ll}=sortedUnitsS;%now just assigning it 
                    semiFinalSortedUnitsLabels.data{1, ii}.sortedSpikesLabels{1,ll}=sortedUnitsLabels;%again assignememt
                    accumResult=size(sortedUnitsS); %tell me, how many units are found at each electrode
                    accumResult=accumResult(2); % calculating number of sorted_units
                    accumulativeActivityPerUnit = zeros(1,accumResult);%just preallocation
                    for mm=1:1:accumResult
                        accumulativeActivityPerUnit(mm)=sum(sortedUnitsS(:,mm));%adding the spiking activity of each unit at particular
                        %electrode during one trial for some specefic
                        %events
                    end
                    finalSortedUnitsResult.data{1, ii}.sortedSpikesUnit{1,ll}=accumulativeActivityPerUnit;
                    if ~isempty(finalSortedUnitsResult.data{1, ii}.sortedSpikesUnit{1,ll})
                        accumulativeActivityUnitsResult.data{1, ii}.sortedSpikesUnit{1,ll}=accumulativeActivityPerUnit;
                    end
                    
                end
                
            end
            
        end
    end
    methods(Static,Access=private)  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %specifically,Functions Used by extractSortedDataFeatures functions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        function spikingActivityUnitsEachTask =sortedUnitsPerTask(sortedDataSession,eventsTask,numberChannels,startingIndex,startingOffset,endingIndex,endingOffset)            
            
            %Explanation: This function will take
            %sortedDataSession,eventsTask,numberChannels,staringIndex,startingIndex,
            %startingOffset, endingIndex, endingOffset as input argumennt
            %and will give spiking activity at each electrode from specefic
            %units during specfic time(events)  for one trials at a time 
            %as output in the form of struct as output argument 
            
            %InputParmeters:
            %Data type of the parameter are well explained at function from
            %which this function is called
            
            %OutputParameter:
            %spikingActivityUnitsEachTask: this of datatype struct and it 
            %has two more struct as fields and there
                %names are 
                % sortedSpikesUnit
                % sorttedUnitsIds
                
                %sortedSpikesUnit: contain 1*m cells, where m is total
                %number of implanted electrodes and every cells is 1*n list
                %and it each entry tells me spiking activity of each unit for one trial 
                %during the the specefic events
                %sorttedUnitsIds: contain 1*m cells, where m is total
                %number of implanted electrodes and every cells is 1*n list
                %and it each entry tells me id of each unit.
            
            data=sortedDataSession;
            eventTime=eventsTask;
            %total_number_of_events = length(event_time);
            totalNumnberOfChannels = numberChannels;
            
            for kk=1:1:totalNumnberOfChannels % for each electrode
                units= data.information{1,kk}(:,1); %sorted units at each elecrode 
                %spike_indeces_kk=data.information{1,kk}(:,2); % spike_indices
                timeStamps=data.information{1,kk}(:,3);  % time stamps of spikes at each electrode
                sortedUnits = unique(units); % total number of sorted units at each electrode
                totalSortedUnits = length(sortedUnits);
                spikeActivityOverCompleteTask = zeros(1,totalSortedUnits); %spike_activity_over_complete_task = number_of_events,total_sorted_units
                %Just using it here to preallocate the space in memory
                activityParticularUnitPerEvent=zeros(totalSortedUnits,1);
                %time_stamp_per_event= zeros(total_number_of_events,2);
                lowerLimit=eventTime(startingIndex)/3.3333e-05 + startingOffset/3.3333e-05;
                upperLimit= eventTime(endingIndex)/3.3333e-05 + endingOffset/3.3333e-05;
                spikeActivityIndex = find(timeStamps>=lowerLimit & timeStamps < upperLimit) ; %index of the unit
                spikeActivityValues =units(spikeActivityIndex);  % identified units of particular electrode
                for jj =1:1:totalSortedUnits    %% at each electrode, for each and every unit
                    activityParticularUnitPerEvent(jj)= sum(spikeActivityValues==sortedUnits(jj));%suming the spiking activity
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                spikeActivityOverCompleteTask(1,:)=activityParticularUnitPerEvent; %%%% most important when spiking activity
                                                                            %has to be claucated between two events and not more than that.
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                spikingActivityUnitsEachTask.sortedSpikesUnit{kk}=spikeActivityOverCompleteTask; 
                spikingActivityUnitsEachTask.sortedUnitsIds{kk}=sortedUnits;
            end
        end
        
        
        
    end %end
    
    
    
end



