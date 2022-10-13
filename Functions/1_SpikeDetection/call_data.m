%% --- Header
    %Author:        A. Erbslöh
    %Company:       UDE-EBS
    %Created on:    20.04.2022
    %Changed on:    18.07.2022    
    %Version:       0v2
    %Project:       Modul zum Abrufen der Neurodaten
    %               inkl. Interpolation und Denoising
    %Comments:      

function [Uin, Fs, Labeling, DataPath, FileName] = call_data(Type, desiredFs, TRange)
    LinkDataPath = '2_Data';
    LinkFileType = {'*.mat; *.edf; *.dat; *.bin', 'Neural datasets (.mat, .edf, .dat; .bin)'; ...
        '*.*', 'All files (*.*)'};
    
    t0 = [];    Uin0 = {};
    DataType = 0;
    datum = datetime('today');
    Labeling = struct();
    
    %% --- Vorauswahl bei der Datenselektion
    if(~Type)
        % --- Aufrufen von Neurodaten
        disp('... calling data sets');
        [FileName, DataPath] = uigetfile(LinkFileType, 'Select a neural dataset', LinkDataPath, 'MultiSelect', 'on');
        % --- Auswahl der Datenquelle
        FileString = split(DataPath(1:end-1), '\');
        DataType = 1 + hex2dec(FileString{end}(1:2)); 
        save('config_data.mat', 'DataPath', 'FileName', 'DataType', 'datum');
    else
        % --- Aufruf der alten Konfiguration
        disp('... loading old config');
        load('config_data.mat');
    end
    %% --- Aufruf von Daten
    switch(DataType)
        case 0 
            % Durchgang für andere Analysen (hier nicht implementiert)
        case 1
            %% --- Datengenerator (Eigenentwicklung, not ready)
            disp(['... load dataset ', num2str(DataType),': Eigener Datengenerator (Prototyp)']);
            disp(['... load data: ', FileName]);
            load([DataPath FileName]);
                                     
            % --- Metadaten
            origFs = SampleRate;
            GainPre = 10^(32/20);
            NoElectrodes = 1;
            TypeMEA = 'Synthetic';

            Labeling.Exists = 1;
            Labeling.OrigClusterID = GroundTruth(2,:);
            Labeling.OrigNoCluster = length(unique(GroundTruth(2,:)));
            Labeling.OrigXPosSpike = GroundTruth(1,:);
            Labeling.OrigNoSpikes = length(GroundTruth(1,:));
        
            % --- Datenverarbeitung
            Uin0{1} = U_EL/GainPre;
            t0 = (0:1:length(U_EL)-1)/origFs;
        case 2 
            %% --- Simulationsdatensätze (Martinez et al, J Neurosci Methods, 2009)
            disp(['... load dataset ', num2str(DataType),': Simulierte Datensätze (Martinez, 2009)']);
            disp(['... load data: ', FileName]);
            load([DataPath FileName]);           
            
            % --- Metadaten
            origFs = 1e3/samplingInterval;
            GainPre = 10^(0/20);                  
            NoElectrodes = 1;
            TypeMEA = 'Synthetic';
            Labeling.Exists = 1;
            Labeling.OrigClusterID = spike_class{1};
            Labeling.OrigNoCluster = length(unique(spike_class{1}));
            Labeling.OrigXPosSpike = spike_times{1};
            Labeling.OrigNoSpikes = length(spike_times{1});
            
            % --- Datenverarbeitung
            Uin0{1} = 0.5e-6*data/GainPre;
            t0 = (0:1:length(data)-1)/origFs;
        case 3 
            %% --- Simulationsdatensätze (Pedreira et al, J Neurosci Methods, 2012) 
            disp(['... load dataset ', num2str(DataType),': Simulierte Datensätze (Pedreira, 2012)']);
            disp(['... load data: ', FileName]);
            load([DataPath FileName]);
            load([DataPath 'ground_truth.mat']);
            dataNoCell = strsplit(FileName(1:end-4), {'_'; '.'});
            dataNo = str2double(dataNoCell{2});
                       
            origFs = 24e3;
            GainPre = 10^(0/20);   
            NoElectrodes = 1;
            TypeMEA = 'Synthetic';
           
            Uin0{1} = 25e-6*data/GainPre;
            t0 = (0:1:length(data)-1)/origFs;
            
            Labeling.Exists = 1;
            Labeling.OrigClusterID = spike_classes{dataNo};
            Labeling.OrigNoCluster = length(unique(spike_classes{dataNo}));
            Labeling.OrigXPosSpike = spike_first_sample{dataNo};
            Labeling.OrigNoSpikes = length(spike_first_sample{dataNo});
        case 4
            %% --- Simulationsdatensätze (Benchmark, Quiroga et al, UCL, 2020)
            disp(['... load dataset ', num2str(DataType),': Simulierte Datensätze']);
            load([DataPath FileName]);
            
            origFs = 1e3/samplingInterval;
            GainPre = 10^(0/20);
            TypeMEA = 'Synthetic';
            
            Uin0{1} = data/GainPre;
            t0 = (0:1:length(data)-1)/origFs;
            
            NoElectrodes = 1;
            Labeling.Exists = 1;
            Labeling.OrigClusterID = spike_class{1};
            Labeling.OrigNoCluster = length(unique(spike_class{1}));
            Labeling.OrigXPosSpike = spike_times{1};
            Labeling.OrigNoSpikes = length(spike_times{1});
        case 5
            %% --- Messdaten aus Promotion Seidl mit penetrierenden MEA (Tiefenhirn)
            disp(['... load dataset ', num2str(DataType),': Recordings from PhD time of Karsten Seidl']);
            disp(['... load data: ', FileName]);
            DataInfo = edfinfo([DataPath FileName]);
            DataSet = edfread([DataPath FileName]);  
            
            GainPre = 10^(52/20);
            NoElectrodes = size(DataSet,2);
            origFs = DataSet.Properties.SampleRate* length(cell2mat(DataSet{1,1})'); 
            TypeMEA = 'BrainTool_IMTEK';
            
            dX = 2;
            for idx= 1:1:NoElectrodes
                Input = cell2mat(table2array(DataSet(1+dX:end-dX,idx)));
                Uin0{idx} = transpose(Input)/GainPre; 
            end
            t0 = (0:1:length(Uin0{1})-1)/origFs;
            
            Labeling.Exists = 0;
        case 6
            %% --- Datensatz aus Neurodatenbank CRCNS-HC1
            disp(['... load dataset ', num2str(DataType),': Neurodatenbank CRCNS-HC1']);
            disp(['... load data: ', FileName]);
            % Check if mat-File is available
            % exist([DataPath FileName], 'file')

            % --- Read DAT File
            MetaData = readstruct([DataPath FileName(1:end-3) 'xml']);
        
            NoElectrodes = MetaData.acquisitionSystem.nChannels;
            origFs = MetaData.acquisitionSystem.samplingRate;
            GainPre = MetaData.acquisitionSystem.amplification;
            Offset = MetaData.acquisitionSystem.offset;
            TypeMEA = 'unknown';

            fID = fopen([DataPath FileName], 'r');
            DataSet = fread(fID, 'uint16') - Offset;
            fclose(fID);
            
            for idx = 1:1:NoElectrodes
                for idx = 0:1:length(DataSet)/NoElectrodes-1
                    Uin0{idx}(idx+1) = DataSet(idx + idx*NoElectrodes)/GainPre;
                end
            end
            t0 = (0:1:length(Uin0{1})-1)./origFs;
                            
            Labeling.Exists = 0;
        case 7
            %% --- Datensatz von Extracellular recordings of Ganglion Cells in mice retina
            disp(['... load dataset ', num2str(DataType),': Ganglion cells, mice retina (Marre, 2018)']);
            load([DataPath FileName(1:end-3) 'mat']);
            
            origFs = fs;
            GainPre = 1;
            
            % --- Determine number of electrodes
            if(0)
                TakeElectrode = [15 33 68 134 141 143 146 232];
            else
                TakeElectrode = 1:size(data,2);
            end
                
            TypeMEA = 'unknown';
            NoElectrodes = length(TakeElectrode);
            for idx = 1:NoElectrodes
                Uin0{idx} = 1e-6 *Gain *(double(data(:,TakeElectrode(idx))) -2^15 +1)';
            end
            t0 = (0:1:length(Uin0{1})-1)./origFs;
            
            Labeling.Exists = 0;
        case 8
            %% --- Recordings with MCS-60MEA from FZJ IBI-1 on rd10 retinae
            disp(['... load dataset ', num2str(DataType),': Recordings from FZJ IBI-1 (Müller), rd10 retinae']);
            load([DataPath FileName]);
            
            if(istable(data))
                Input = table2array(data(:, 2:end));                
            else
                Input = data;
            end
            
            NoElectrodes = size(Input, 2);
            origFs = Fs;
            GainPre = 1;
            TypeMEA = 'MCS 60MEA';

            for idx = 1:NoElectrodes
                Uin0{idx} = 1e-6* transpose(Input(:, idx));
            end
            t0 = (0:1:length(Uin0{1})-1)/Fs;
         
            Labeling.Exists = 0;
        case 9
            %% --- Recordings with Shank-BiMEA from FZJ IBI-3 on rd10 and wt retinae
            disp(['... load dataset ', num2str(DataType),': Recordings from FZJ IBI-3 (Montes), rd10 and wt retinae']);
            load([DataPath FileName]);

            origFs = Settings.Fs;
            GainPre = 1;
            NoElectrodes = size(data, 2);
            TypeMEA = 'MCS 60MEA';
            
            for idx = 1:NoElectrodes
                Uin0{idx} = data(:,idx)/Settings.Gain;
            end
            t0 = (0:1:size(data, 1))./origFs;
            
            Labeling.Exists = 0;
        case 10
            %% --- Datensatz aus Harvard - Neuropixels in Human Cortex (Paulk, 2021)
            disp(['... load dataset ', num2str(DataType),': Neuropixels (Human Cortex, Paulk, 2021)']);
            [filepath, filename, format] = fileparts([DataPath FileName]);
            
            % --- Read spike data
            origFs = 30e3;
            GainPre = 1;
            TypeMEA = 'NeuroPixel 1.0';

            DataAP_Origin = strcat(filepath, '\', filename(1:end-2), 'ap', format);
            fid = fopen(DataAP_Origin, 'r');
            DataAP = fread(fid, [385 Inf], '*int16');
            fclose(fid);

            chanMap = readNPY('channel_map.npy');
            dat = dat(chanMap+1,:);

            % --- Read LFP data
            if(0)
                origFs_LF = 2.5e3;
                DataLFP_Origin = strcat(filepath, '\', filename(1:end-2), 'lf', format);
                fid = fopen(DataLFP_Origin, 'r');
                DataLFP = fread(fid, [385,Inf], 'int16=>single');
                fclose(fid);
            end

            NoElectrodes = size(DataAP, 1);
            for idx = 1:NoElectrodes
                Uin0{idx} = 1e-6* double(DataAP(idx,:));
            end            
            t0 = (0:1:size(DataAP,2)-1)./origFs;
            origFs = origFs;

            Labeling.Exists = 0;
        case 11
            %% --- Datensatz vom CortexLab - Neuropixels
            disp(['... load dataset ', num2str(DataType),': CortexLab with Neuropixels (xyz)']);
                        
            % --- Read spike data
            origFs = 30e3;
            GainPre = 1;
            TypeMEA = 'NeuroPixel 1.0';

            fid = fopen([DataPath FileName], 'r');
            DataAP = fread(fid, [385 Inf], '*int16');
            fclose(fid);   

            NoElectrodes = size(DataAP, 1);
            for idx = 1:NoElectrodes
                Uin0{idx} = 1e-6* double(DataAP(idx,:));
            end            
            t0 = (0:1:size(DataAP,2)-1)./origFs;
            origFs = origFs;

            Labeling.Exists = 0;
        case 170
            %% --- Test
            disp(['... load dataset ', num2str(DataType),': Content description']);
            
            % --- Selection if big file is selected or small packets
            if size(FileName,2) > 1
                TextLength = [1 Inf];
                for idx = 1:size(FileName,2)
                    fid = fopen([DataPath FileName{idx}], 'r');
                    DataAP(idx,:) = fread(fid, TextLength, '*int16');
                    fclose(fid); 
                end
            else
                TextLength = [385 Inf];
                fid = fopen([DataPath FileName], 'r');
                DataAP = fread(fid, TextLength, '*int16');
                fclose(fid);   
            end    
              
            origFs = 30e3;
            GainPre = 1;
            NoElectrodes = size(DataAP,1);
            TypeMEA = 'NeuroPixel 1.0';
            
            for idx = 1:NoElectrodes
                Uin0{idx} = 1e-6*double(DataAP(idx,:));
            end
            t0 = (0:1:size(DataAP,2)-1)/origFs;
         
            Labeling.Exists = 0;
        
        case 256
            %% --- Template
            disp(['... load dataset ', num2str(DataType),': Content description']);
            load([DataPath FileName]);
                       
            t0 = Input(:,1)';
         
            Labeling.Exists = 0;
        otherwise
            error('Loading data set failed! Please try again!');        
    end

    %% --- Datenmenge auf Zeitbereich reduzieren
    LengthDataOrig = length(t0);
    if(length(TRange) == 2)
        T0 = find(TRange(1) <= t0, 1);
        T1 = find(TRange(2) <= t0, 1);
        t0 = t0(T0:T1);
        for idx = 1:1:NoElectrodes
            Uin0{idx} = Uin0{idx}(T0:T1);
        end
    else
        T0 = 1;
        T1 = length(t0);
    end
    
    %% --- Resampling data
    LengthDataCut = length(t0);
    if(desiredFs ~= origFs && desiredFs ~= 0)
        Uin = cell(size(Uin0));
        t = zeros(size(t0));
        % Checking if mean value of the signal is zero
        for idx = 1:1:NoElectrodes
            Usafe = 5e-6;
            if(abs(sum((mean(Uin0{idx})-[-1 1]*Usafe) < 0)-1) == 1)
              dU = mean(Uin0{idx});
            else 
              dU = 0;
            end
            [Uin{idx}, t] = resample(Uin0{idx}-dU, t0, desiredFs);
            Uin{idx} = Uin{idx} + dU;
            %Uin{k} = interp(Uin0{k}, desiredFs/origFs);
        end
        Fs = desiredFs;
    else
       Uin = Uin0;
       t = t0;
       Fs = origFs;
    end

    %% --- Ausgabe von Meta-Informationen
    disp(['... sampling rate of ', num2str(origFs/1e3), ' kHz']);
    disp(['... using ', num2str(round(LengthDataCut/LengthDataOrig *100, 2)), '% content of the data']);
    disp(['... data includes an array of ', num2str(NoElectrodes), ' electrodes (', TypeMEA, ')']);
    
    %% --- Anpassung und Ausgabe des Labeling    
    if(~Labeling.Exists)
        disp('... dataset includes no labeling'); 
    else
        X0 = find(Labeling.OrigXPosSpike -T0 >= 0, 1, 'first');
        X1 = find(Labeling.OrigXPosSpike -T1 <= 0, 1, 'last');
    
        Labeling.IstClusterID = Labeling.OrigClusterID(X0:X1);
        Labeling.IstNoCluster = length(unique(Labeling.IstClusterID));
        if(X0 > 1)
            PreviousValue = Labeling.OrigXPosSpike(X0-1);
        else
            PreviousValue = 0;
        end
        Labeling.IstXPosSpike = round((Labeling.OrigXPosSpike(X0:X1) - T0)* desiredFs/origFs);
        Labeling.IstNoSpikes = length(Labeling.IstXPosSpike);
        
        disp(['... data includes labeling (NoCluster = ', num2str(Labeling.IstNoCluster), ...
            ', NoSpike = ', num2str(Labeling.IstNoSpikes), ')']);
    end
end