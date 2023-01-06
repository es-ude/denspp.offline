close all;  clear all;  clc;
warning('off');
opengl('save', 'hardwarebasic');

%% --- Header
    %Author:        J. Kuhn
    %Company:       UDE-EBS
    %Created on:    30.06.2022
    %Changed on:    12.07.2022    
    %Version:       0v2
    %Project:       Anwendung zum Labeln von Spike-Neurodatensätzen
    %Commments:      
    
disp('---- Application for labeling neural spiking datasets ----');

%% --- Pre-defined settings
SkriptVersion = '0v2';
DateToday = datetime();
GroundTruth = [];

%% --- Read data input
addpath('Functions', 'Functions');

SaveGroundTruth = 0;
% Auswahl des zu verwendenen Datenquelle
% (0: Datensatz, 1: Alte Konfig.)
SetDataType = 0;
% Angabe des zu betrachteten Zeitfensters [Start Ende] in sec.
% (TRange = 0 ruft vollen Datensatz auf)
SetTRange = 0; %[0 20];

[Uin, SampleRate, ~, DataPath, FileName] = call_data(SetDataType, 0, SetTRange);

%% --- Settings for pipeline
% --- Filtering
AFE_SET.SampleRate = SampleRate;
AFE_SET.fFilt_ANA = [200 5e3];
% --- Time delay
AFE_SET.InputDelay = round(1e-3* SampleRate);
% --- Properties of spike detection
AFE_SET.dXsda = 2;      
AFE_SET.ThresMode = 3;
AFE_SET.SDA_ThrMin = 1.5e-9; %(only for mode=1)
% --- Properties of Framing and Aligning of spike frames
AFE_SET.XDeltaNeg = round(0.4e-3* SampleRate);
AFE_SET.XWindowLength = round(1.6e-3* SampleRate);
AFE_SET.XOffset = round(0.4e-3* SampleRate);
AFE_SET.XDeltaPos = AFE_SET.XWindowLength - AFE_SET.XDeltaNeg;
AFE_SET.NoCluster = 3;

%% --- Filtering and Denoising input signal
SelCH = 1;

% --- Filtering input signal
[bLFP, aLFP] = butter(2, 2*[1e-2 AFE_SET.fFilt_ANA(1)]/SampleRate, 'bandpass');
[bAP, aAP] = butter(2, 2*AFE_SET.fFilt_ANA/SampleRate, 'bandpass');
Ulfp = {};
Uspike = {};
for idx = 1:size(Uin,2)
    Ulfp{idx} = filtfilt(bLFP, aLFP, Uin{idx});
    Uspike{idx} = filtfilt(bAP, aAP, Uin{idx});
    
    % --- Determining range
    MinA = 1e6*min(Uspike{idx});
    MaxA = 1e6*max(Uspike{idx});

    Range(idx,1) = sign(MinA)* ceil(abs(MinA)/ 10^floor(log10(abs(MinA))))* 10^floor(log10(abs(MinA)));
    Range(idx,2) = sign(MaxA)* ceil(abs(MaxA)/ 10^floor(log10(abs(MaxA))))* 10^floor(log10(abs(MaxA)));
end

Uspk = Uspike{SelCH};

%% --- ToDo: Denoising signal and checking if activity is available
WindowLength = floor(2e-3* SampleRate);
WindowLength = (1-mod(WindowLength,2)) + WindowLength;

method = 1;
% --- Denoising input signal Uspk - output: Utest
switch(method)
    case 0
        % Zeitlicher Mittelwert
        Utest = movmean(Uspk, floor(2.5* WindowLength));
    case 1
        % Golay-Filter
        Utest = sgolayfilt(Uspk, 3, WindowLength);
    case 2
        % Wavelet-Transformation mit sym4-Wavelet
        Utest = wdenoise(Uspk, 2, 'Wavelet', 'sym4', 'DenoisingMethod', 'BlockJS');
    case 3
        % Kommando 1
        % Kommando 2
        Utest = 1;
end

% --- Check if activity is available
ProoveNoise = [min(Utest), mean(Utest), std(Utest), max(Utest)];  

%% --- Spike Detection and Feature Extraction
[Xtrg, Usda, Uthr] = SpikeDetection(AFE_SET, Uspk, AFE_SET.ThresMode);
try
    [FramesOrig, XPos] = FrameGeneration(AFE_SET, Uspk, Xtrg);
    FramesAlign = FrameAligning(AFE_SET, FramesOrig, 2);  
    
    [~, PCA_score] = pca(FramesAlign);
    FeaturesArray = PCA_score(:,1:end)';
    ClusterID = transpose(kmeans(FeaturesArray', AFE_SET.NoCluster));
    
    SpikeTicks = DetermineSpikeTicks(t, XPos, ClusterID, AFE_SET.NoCluster);

    runSuccess = 1;
    disp("... running pipeline successful")
catch
    disp("... running pipeline with error");
    runSuccess = 0;
end

%% --- Determining and saving groundtruth
GroundTruth0 = [];
if(SaveGroundTruth && runSuccess)
    for idx = 1:100
        GroundTruth0(idx) = idx;
    end    
    save([DataPath, '\', FileName(1:end-4), '_AddedLabeling', '.mat'], 'SkriptVersion', 'DateToday', 'GroundTruth0');
end

%% --- Plotten
close all;

t = (0:1:length(Uspk)-1)/SampleRate;

plotResults_InMEA(t, Ulfp, Uspike, Range);
%plotResults_InMEA2(t, Ulfp{SelCH}, Uspike{SelCH}, SpikeTicks, Range);
if(runSuccess && 0)
    plotResults_Labeling(t, Uspk, Usda, Uthr, XPos, FramesOrig, FramesAlign, FeaturesArray, ClusterID, SpikeTicks, GroundTruth0, 0);
end
disp('... Done');

%% --- Functions for Labeling
% --- Spike Detection       
function [Xtrg, Xsda, Xthr] = SpikeDetection(Setting, Xin, mode)
    % --- Einfache Variante für direkte Anwendung
    % Xsda = Xin(k+1:end-k).^2 - Xin(1:end-2*k).* Xin(2*k+1:end); 
    % [YPks, XPks] = findpeaks(Xsda, 'MinPeakHeight', Settings.ThresholdSDA, 'MinPeakDistance', round(500e-6*SampleRate)); 
    % --- Methoden
    % Auswahl erfolgt über die Vektor-Länge von dXsda 
    % length(x) == 1:   mit dX = 1 --> NEO, dX > 1 --> k-NEO
    % length(x) > 1:    M-TEO
        
    k = Setting.dXsda;
    Xsda = Xin(k+1:end-k).^2 - Xin(1:end-2*k).* Xin(2*k+1:end);
    Xsda = [ones(1,k)*Xsda(1) Xsda ones(1,k)*Xsda(end)];

    % --- Schwellenwert-Bestimmung
    Xthr = Thres(Setting, Xsda, mode);
    
    % --- Trigger-Generierung
    Xtrg = double(Xsda >= Xthr);
end    
% --- Threshold determination for neural input
function Xout = Thres(Setting, Xin, mode)
    WindowLength = floor(2e-3* Setting.SampleRate);
    WindowLength = 1- mod(WindowLength, 2) + WindowLength;
    switch(mode)
        case 1 % Constant value
            Xout = Setting.ThrMin;
        case 2 % Standard derivation of background activity
            Xout = 4* mean(abs(Xin)/0.6745);
        case 3 % Automated calculation of threshold (using by BlackRock)
            Xout = 4.5* sqrt(sum(Xin.^2)/length(Xin));
        case 4 % Mean value 
            Xout = 5* mean(Setting.MemSDA);
        case 5 % Lossy peak detection
            Xout = envelope(Xin, WindowLength, 'rms');  
        case 6 % movmean
            Xout = 2* movmean(Xin, WindowLength);
        case 7 % window mean method for max-detection
            Xout = 0*Xin;
            for i = 1:1:floor(length(Xin)/WindowLength)
               X0 = [1 WindowLength] + (i-1)*WindowLength;
                Xout(X0(1):X0(2)) = max(Xin(X0(1):X0(2))); 
            end
            Xout = 10*movmean(Xout, 200); %mean(Xhi, 100);
    end
    % --- Transfer array size to time vector
    if(length(Xout) == 1)
        Xout = 0* Xin + Xout;
    else 
        Xout = Xout;
    end
end

% --- Frame Generation
function [Frame, Xpos] = FrameGeneration(Setting, Xin, Xtrg)
    % --- Check if no results are available
    if(sum(Xtrg) == 0)
        Frame = [];
        Xpos = [];
        return;
    end
    % --- Extract x-positions from the trigger signal
    %Xpos = 1 + find(diff(Xtrg) == 1);
    
    [~, Xpos] = findpeaks(Xtrg, 'MinPeakDistance', Setting.XWindowLength, 'MinPeakHeight', 0.7);
    % --- Extract frames
    Frame = [];
    for i = 1:1:length(Xpos)
        dXneg = Xpos(i) - Setting.XDeltaNeg - Setting.XOffset;
        dXpos = Xpos(i) + Setting.XDeltaPos + Setting.XOffset -1;
        Frame(i,:) = Xin(dXneg:dXpos); 
    end
end
% --- Frame Aligning
function FrameOut = FrameAligning(Setting, FrameIn, AlignMode)    
    FrameOut = [];
    % --- Check if no results are available
    if(isempty(FrameIn))
        return;
    end
    % --- Align each frame to specific point
    for x = 1:1:size(FrameIn, 1)
        Frame0 = FrameIn(x,:);
        Frame = movmean(Frame0, 2);
        switch(AlignMode)
            case 1  % Maximum Aligning
                [~, max_pos] = max(Frame);
            case 2  % Aligned to positive turning point
                [~, max_pos] = max(diff(Frame)); 
                max_pos = max_pos +1;
            case 3  % Aligned to negative turning point
                [~, max_pos] = min(diff(Frame)); 
                max_pos = max_pos +1;
        end
        
        Xpos0(x,:) = max_pos + [-Setting.XDeltaNeg 0 +Setting.XDeltaPos]; 
        Xpos = Xpos0(x,:);
        if(Xpos(3) > length(Frame0))
            state = 1;
        elseif(Xpos(1) <= 0)
            state = 2;
        else
            state = 0;
        end
        switch(state)
            case 0
                FrameOut(x,:) = Frame0(Xpos(1) : Xpos(3)-1);
            case 1
                FrameOut(x,:) = [Frame0(Xpos(1):end) Frame0(end)*ones(1, abs(Xpos(3)-length(Frame0))-1)];
            case 2
                FrameOut(x,:) = [Frame0(1)*ones(1, abs(Xpos(1))) Frame0(1:Xpos(3))];                      
        end
    end
end
% --- Function for determining spike ticks
function SpikeTicks = DetermineSpikeTicks(Time, XPos, ClusterIn, NoCluster)
    SpikeTicks = zeros(NoCluster, length(Time));
    for i = 1:1:NoCluster
        XCluster = find(ClusterIn == i);
        SpikeTicks(i, XPos(XCluster)) = 1;                    
    end
end