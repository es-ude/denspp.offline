close all;  clear all;  clc;
warning('off');
opengl('save', 'hardwarebasic');

%% --- Header
    %Author:        J. Kuhn
    %Company:       UDE-EBS
    %Created on:    30.06.2022
    %Changed on:    04.10.2022    
    %Version:       0v2
    %Project:       Anwendung zum Labeln von Spike-Neurodatensätzen
    %Commments:     Toolbox ,,Wavelet'' von Matlab erforderlich 
    
disp('---- Application for labeling neural spiking datasets ----');

%% --- Pre-defined settings
SkriptVersion = '0v2';
DateToday = datetime();
GroundTruth = [];

%% --- Read data input
addpath('Functions')
SaveGroundTruth = 0;
% Auswahl des zu verwendenen Datenquelle
% (0: Datensatz, 1: Alte Konfig.)
SetDataType = 0;
% Angabe des zu betrachteten Zeitfensters [Start Ende] in sec.
% (TRange = 0 ruft vollen Datensatz auf)
SetTRange = 0; %[0 20];

[Uin, SampleRate, ~, DataPath, FileName] = call_data(SetDataType, 0, SetTRange);


%% --- Filtering and Denoising input signal
SelCH = 1;

% --- Filtering input signal
[bLFP, aLFP] = butter(2, 2*[1e-2 100]/SampleRate, 'bandpass');
[bAP, aAP] = butter(2, 2*[100 5e3]/SampleRate, 'bandpass');
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

method = 10;
% --- Denoising input signal
switch(method)
    case 0
        Utest = movmean(Uspk, floor(2.5* WindowLength));
    case 1
        Utest = sgolayfilt(Uspk, 3, WindowLength);
    case 2
        Utest = wdenoise(Uspk, 2, 'Wavelet', 'sym4', 'DenoisingMethod', 'BlockJS');
    case 3 
        Utest = cmddenoise (Uspk,'bior2.2',9, 's') ;
    case 4
        Utest = cmddenoise (Uspk, 'sym4',1.0, 'h',2 );
    case 5 % !!! sehr lange Durchlaufzeit 
        wv = 'db6';  
        Utest = wavedec2 (Uspk, 8, wv); 
   case 6 
        wv = 'db10';
        thr = 100;
        sorh = 's';
        keepapp = 0;
        Utest = wdencmp ('gbl', Uspk,wv,8,thr,sorh,keepapp);
    case 7
        Utest = imtophat(Uspk,1); 
    case 8
        Utest = imbothat (Uspk,1);
    case 9
        Utest = smoothdata (Uspk,2,'gaussian', WindowLength);
    case 10
        Utest = smoothdata (Uspk,1,'sgolay', WindowLength);
    case 11 % !!! sehr lange Durchlaufzeit (>20min) 
       [Utest, f] = wsst(Uspk,1,'bump'); 
    case 12 % braucht zu lange zum Durchlaufen und hängt sich auf
        Utest = cwt (Uspk, 'amor', minutes(1/60));
    case 13 % braucht zu lange zum Durchlaufen und hängt sich auf
        Utest = cwtfilterbank (WindowLength , 'bump');  
    case 14
        fb = cwtfilterbank;
        Utest = wt (fb,Uspk);

    end
       
% --- Check if activity is available
ProoveNoise = [min(Utest), mean(Utest), std(Utest), max(Utest)];  


%% --- Plotten
close all;
runSuccess = 0;
t = (0:1:length(Uspk)-1)/SampleRate;

plotResults_InMEA(t, Ulfp, Uspike, Range);
%plotResults_InMEA2(t, Ulfp{SelCH}, Uspike{SelCH}, SpikeTicks, Range);
if(runSuccess && 0)
    plotResults_Labeling(t, Uspk, Usda, Uthr, XPos, FramesOrig, FramesAlign, FeaturesArray, ClusterID, SpikeTicks, GroundTruth0, 0);
end
disp('... Done');
