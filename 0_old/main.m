close all;  clear all;  clc;
warning('off');

%% --- Header
    %Author:        A. Erbslöh
    %Company:       UDE-EBS
    %Created on:    20.02.2021
    %Changed on:    19.07.2022    
    %Version:       1v1
    %Project:       Modell zum Entwurf eines Spike Sorters
    %Commments:     Only for designing analogue front-end
    %Necesary toolboxes:    Signal processing Toolbox, Machine Learning Toolbox, Parallel Computing Toolbox                  
    
disp('---- MATLAB-Environment for Spike Sorting ----');

%% --- Read settings
run("config.m");

%% --- Read data input
addpath('Functions/0_Plotting', 'Functions/1_SpikeDetection', 'Functions/2_SpikeSorting', 'Functions/3_Decoder');

[AFE_Signals.Uin, Fs, GroundTruth] = call_data(Settings.DataType, Settings.DesiredFs, Settings.TRange);
Settings.CH_TotNo = size(AFE_Signals.Uin, 2);

%% --- Preparation: Module calling 
AFE = afe(AFE_SET, Fs, Settings.RealtimeMode);

%% --- Preparation: Variable declaration
AFE_Signals.Ulfp = cell(1, length(Settings.CHsel));
AFE_Signals.Uspk = cell(1, length(Settings.CHsel));

AFE_Signals.Xadc = cell(1, length(Settings.CHsel));
AFE_Signals.Xsda = cell(1, length(Settings.CHsel));
AFE_Signals.Xthr = cell(1, length(Settings.CHsel));
AFE_Signals.Xtrg = cell(1, length(Settings.CHsel));

AFE_Signals.XPos        = cell(1, length(Settings.CHsel));
AFE_Signals.FramesOrig  = cell(1, length(Settings.CHsel));
AFE_Signals.FramesAlign = cell(1, length(Settings.CHsel));
AFE_Signals.Features    = cell(1, length(Settings.CHsel));
AFE_Signals.ClusterID   = cell(1, length(Settings.CHsel));
AFE_Signals.SpikeTicks  = cell(1, length(Settings.CHsel));

%% --- Calculation
if(Settings.CHsel == 0)
    Settings.CHsel = 1:Settings.CH_TotNo;
end

disp('... run SpikeSorting');
for idx = 1:1:length(Settings.CHsel)
    idxCH = Settings.CHsel(idx);

    % --- Anpassungen für Realtime-Anwendung
    doADC = 1;
    Uin = AFE_Signals.Uin{idxCH};
        
    % --- Modules of Analogue Front-end
    [Uspk, Ulfp] = AFE.PreAmp(Uin);
    [Xadc, ~] = AFE.ADC_Nyquist(Uspk, doADC); 
    Xdly = AFE.TimeDelay_DIG(Xadc);
    %Xfilt = AFE.DigFilt(Xadc, doADC);    
    [Xtrg, Xsda, Xthr] = AFE.SpikeDetection(Xadc, AFE_SET.ThresMode, doADC);
    [FramesOrig, XPos] = AFE.FrameGeneration(Xdly, Xtrg);
    FramesAlign = AFE.FrameAligning(FramesOrig, 2);       

    % --- Modules of Feature Extraction and Classification 
    % (only for pre-labeling data input)
    [FeatArray, FeatCell] = AFE.FE_PCA(FramesAlign);
    %[FeatArray, FeatCell] = AFE.FE_Normal(FramesAlign);
    ClusterProps = AFE.Clustering(FeatArray);
    SpikeTicks = AFE.DetermineSpikeTicks(XPos, ClusterProps, Xadc); 
    
    % --- After Processing for each channel
    AFE_Signals.Uspk{idxCH} = Uspk;
    AFE_Signals.Ulfp{idxCH} = Ulfp;
    
    AFE_Signals.Xadc{idxCH} = Xadc;
    AFE_Signals.Xsda{idxCH} = Xsda;
    AFE_Signals.Xthr{idxCH} = Xthr;
    AFE_Signals.Xtrg{idxCH} = Xtrg;
    
    AFE_Signals.XPos{idxCH} = XPos;
    AFE_Signals.FramesOrig{idxCH} = FramesOrig;
    AFE_Signals.FramesAlign{idxCH} = FramesAlign;
    AFE_Signals.Features{idxCH} = FeatArray;
    AFE_Signals.ClusterID{idxCH} = ClusterProps;
    AFE_Signals.SpikeTicks{idxCH} = SpikeTicks;
    
    %% --- Determining quality parameters
    if(GroundTruth.Exists)
        GroundTruth.ADCXPosSpike = round(GroundTruth.IstXPosSpike* AFE_SET.SampleRate/Settings.DesiredFs);

        ResultsSDA = AFE.analyzeSDA(Xtrg, GroundTruth.ADCXPosSpike, 100);
        Quality_Param.DR{idxCH} = ResultsSDA.TPR* ResultsSDA.Accuracy;
        Quality_Param.CA{idxCH} = ResultsSDA.Accuracy;
        Quality_Param.CR{idxCH} = length(Uspk)/(size(FramesAlign,1)* size(FramesAlign,2));
    else
        Quality_Param = [];
    end
end

clear idx doADC idxCH;
clear Uin Uspk Ulfp Uadc;
clear Xadc Xdly Xsda Xthr Xtrg;
clear XPos FramesOrig FramesAlign FeatArray FeatCell ClusterProps SpikeTicks;

%% --- Plotten
SelCH = Settings.CHsel(1);
LineWidth = 1;
LineColor = {'k'; 'r'; 'b'; 'g'; 'y'; 'c'; 'm'; 'k'; 'r'; 'b'; 'g'; 'y'; 'c'; 'm'; 'k'; 'r'; 'b'; 'g'; 'y'; 'c'; 'm'};
close all;
disp('... Plotting results');

%plotResults_Threshold(AFE_Signals, [Fs AFE_SET.SampleRate], AFE.LSB, LineColor, LineWidth);
%plotResults_PDAC(AFE_Signals, Quality_Param, SelCH, LineColor, LineWidth);
plotResults_Transient(AFE_Signals, [Fs AFE_SET.SampleRate], AFE.LSB, GroundTruth, SelCH, 0, LineColor, LineWidth);
plotResults_FeatureMap(AFE_Signals, Quality_Param, SelCH, LineColor, LineWidth);
%plotResults_Histo(GroundTruth, AFE_Signals.ClusterID{SelCH});

clear SelCH LineWidth LineColor;
disp('... Done');