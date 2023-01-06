%% --- Header
    %Author:        A. Erbslöh
    %Company:       UDE-EBS
    %Created on:    20.04.2022
    %Changed on:    08.06.2022    
    %Version:       0v2
    %Project:       Module zur analogen Signalvorverarbeitung von 
    %               extrazellulären Aktionspotentials eines MEAs
    %Comments:      /

classdef afe < handle
    %% --- Variablen
    properties(GetAccess = public, SetAccess = private)
        RealtimeMode = 0;
        % --- Power Supply
        UDD = 1.2;
        Ucm = 0.6;
        USS = 0; 
        % --- Variables for pre-amplication
        vPRE = 40;
        nFILT_ANA = 2;
        fRange_ANA = [1 5e3];
        aIIR_ANA;   bIIR_ANA;
        aIIR_LFP;   bIIR_LFP;
        MemIIR_ANA;
        % --- Variables for ADC
        Urange = [0.5 0.7];
        nBitADC = 12;
        LSB;
        partitionADC;
        SampleRateADC;
        SampleRateAna;
        pRatio; qRatio;
        OversamplingRatio;
        OversamplingCNT;
        XoldADC;
        UoldADC;
        % --- Variables for digital filtering
        nFILT_DIG;
        fRange_DIG;
        aIIR_DIG;   bIIR_DIG;
        MemIIR_DIG;
        XoldDIG;
        % --- Variables for Spike detection
        ThrMin = 1e-8;
        dXsda = 1;
        modeSDA = 0;
        MemSDA;
        MemThres;
        XsdaOld;
        % --- Variables for Frame Generation and Aligning
        EnFrame = 0;
        MemFrame;
        InputDelay;
        XDeltaNeg;
        XDeltaPos;
        XOffset;
        XWindowLength;
        % --- Properties for Labeling datasets
        NoCluster = 5;
    end
    % --- Functions for external access
    methods(Access = public)
        %% --- Main function for calling
        function SpikeAFE = afe(Settings, Fs, RealtimeMode)
            SpikeAFE.RealtimeMode = RealtimeMode;
            if exist('Settings.Udd','var')
                SpikeAFE.UDD = 1.2;
            else
                SpikeAFE.UDD = Settings.Udd;
            end
            if exist('Settings.Uss','var')
                SpikeAFE.UDD = 0;
            else
                SpikeAFE.USS = Settings.Uss;
            end
            SpikeAFE.Ucm = (SpikeAFE.UDD + SpikeAFE.USS)/2;
            
            SpikeAFE.vPRE = Settings.GainPre;
            SpikeAFE.nFILT_ANA = Settings.nFilt_ANA;
            SpikeAFE.fRange_ANA = Settings.fFilt_ANA;
            [SpikeAFE.bIIR_ANA, SpikeAFE.aIIR_ANA] = butter(SpikeAFE.nFILT_ANA, 2*SpikeAFE.fRange_ANA/Fs, 'bandpass');
            SpikeAFE.MemIIR_ANA = SpikeAFE.Ucm + zeros(1, length(SpikeAFE.bIIR_ANA)-1);
            
            SpikeAFE.InputDelay = Settings.InputDelay;
            
            SpikeAFE.Urange = SpikeAFE.Ucm + [-1 1]*Settings.dUref;
            SpikeAFE.nBitADC = Settings.nBitADC;
            SpikeAFE.LSB = diff(SpikeAFE.Urange)/2^SpikeAFE.nBitADC;
            SpikeAFE.partitionADC = SpikeAFE.Urange(1)+SpikeAFE.LSB/2 : SpikeAFE.LSB : SpikeAFE.Urange(2)+SpikeAFE.LSB/2;
            SpikeAFE.OversamplingRatio = Settings.Oversampling;
            SpikeAFE.OversamplingCNT = 1;
            SpikeAFE.SampleRateADC = Settings.SampleRate;
            SpikeAFE.SampleRateAna = Fs;
            [SpikeAFE.pRatio, SpikeAFE.qRatio] = rat(SpikeAFE.SampleRateADC* SpikeAFE.OversamplingRatio/SpikeAFE.SampleRateAna);
            
            SpikeAFE.nFILT_DIG = Settings.nFilt_DIG;
            SpikeAFE.fRange_DIG = Settings.fFilt_DIG;
            [SpikeAFE.bIIR_DIG, SpikeAFE.aIIR_DIG] = butter(SpikeAFE.nFILT_DIG, 2*SpikeAFE.fRange_DIG/SpikeAFE.SampleRateADC, 'bandpass');
            [SpikeAFE.bIIR_LFP, SpikeAFE.aIIR_LFP] = butter(SpikeAFE.nFILT_DIG, 2*[1e-2 SpikeAFE.fRange_DIG(1)]/SpikeAFE.SampleRateADC, 'bandpass');
            %SpikeAFE.bIIR_DIG = int16(SpikeAFE.bIIR_DIG/2^(SpikeAFE.nBitADC-1));
            %SpikeAFE.bIIR_DIG = int16(SpikeAFE.bIIR_DIG/2^(SpikeAFE.nBitADC-1));
            SpikeAFE.MemIIR_DIG = 2^(SpikeAFE.nBitADC-1) + zeros(1, length(SpikeAFE.bIIR_DIG)-1);
                        
            SpikeAFE.dXsda = Settings.dXsda;
            SpikeAFE.ThrMin = Settings.SDA_ThrMin;
            if(length(Settings.dXsda) == 1)
                SpikeAFE.modeSDA = 0;
            else
                SpikeAFE.modeSDA = 1;
            end
            SpikeAFE.MemSDA = zeros(1, 2*SpikeAFE.dXsda(end)+1);
            SpikeAFE.MemThres = zeros(1, 100);
            
            SpikeAFE.EnFrame = 0;
            SpikeAFE.XDeltaNeg = Settings.XDeltaNeg;
            SpikeAFE.XWindowLength = Settings.XWindowLength;
            SpikeAFE.XDeltaPos = SpikeAFE.XWindowLength - SpikeAFE.XDeltaNeg;
            SpikeAFE.XOffset = Settings.XOffset;
            SpikeAFE.MemFrame = zeros(1, SpikeAFE.XWindowLength);

            SpikeAFE.NoCluster = Settings.NoCluster;
        end
        %% --- Analogue Preamplifier and Filtering
        function UpdateIIR_ANA(SpikeAFE, Fs)
            [SpikeAFE.bIIR_ANA, SpikeAFE.aIIR_ANA] = butter(SpikeAFE.nFILT_ANA, 2*SpikeAFE.fRange_ANA/Fs, 'bandpass');
            SpikeAFE.MemIIR_ANA = SpikeAFE.Ucm + zeros(1, length(SpikeAFE.bIIR_ANA)-1);
        end
        
        function [Uout, Ulfp] = PreAmp(SpikeAFE, Uin) 
            % Preamplification and Filtering
            if(SpikeAFE.RealtimeMode) 
                U0 = Uin - SpikeAFE.Ucm;
                dU0 = SpikeAFE.aIIR_ANA* [U0 -SpikeAFE.MemIIR_ANA(1,:)]';
                dU1 = SpikeAFE.bIIR_ANA* [dU0 SpikeAFE.MemIIR_ANA(1,:)]';
                SpikeAFE.MemIIR_ANA(1,:) = [dU0 SpikeAFE.MemIIR_ANA(1, 1:end-1)];
                Uout = SpikeAFE.vPRE* dU1 + SpikeAFE.Ucm;
                Ulfp = SpikeAFE.Ucm;
            else
                Uout = SpikeAFE.Ucm + SpikeAFE.vPRE* filtfilt(SpikeAFE.bIIR_ANA, SpikeAFE.aIIR_ANA, Uin-SpikeAFE.Ucm);                 
                Ulfp = SpikeAFE.Ucm + SpikeAFE.vPRE* filtfilt(SpikeAFE.bIIR_LFP, SpikeAFE.aIIR_LFP, Uin-SpikeAFE.Ucm);                 
            end            
            % Voltage clamping
            Uout(Uout >= SpikeAFE.UDD) = SpikeAFE.UDD; 
            Uout(Uout <= SpikeAFE.USS) = SpikeAFE.USS;
        end
        
        %% --- Time delay module
        function Uout = TimeDelay_DIG(SpikeAFE, Uin)
            if isa(Uin, 'int16')
                mat = int16(ones(1, SpikeAFE.InputDelay));
            else
                mat = ones(1, SpikeAFE.InputDelay);
            end
            Uout = [Uin(1)*mat Uin(1:end-SpikeAFE.InputDelay)];
        end
        
        %% --- AD sampling/converting
        function [Xout, Uout] = ADC_Nyquist(SpikeAFE, Uin, EN)
            % Clamping through supply voltage
            UinADC = Uin;
            UinADC(Uin >= SpikeAFE.Urange(2)) = SpikeAFE.Urange(2); 
            UinADC(Uin <= SpikeAFE.Urange(1)) = SpikeAFE.Urange(1);
            
            Uin0 = UinADC(1) + resample(UinADC-UinADC(1), SpikeAFE.pRatio, SpikeAFE.qRatio);
            idx = 1;
            X0 = 1;
            X1 = SpikeAFE.OversamplingRatio;
            while(X0 <= length(Uin0))
                if(SpikeAFE.OversamplingRatio == 1)
                    [Xout(idx), Uout(idx)] = AD_Conv(SpikeAFE, Uin0(idx), 1);
                else
                    if(X1 > length(Uin0))
                        X1 = length(Uin0);
                    end
                    [Xout(idx), Uout(idx)] = AD_Conv(SpikeAFE, mean(Uin0(X0:X1)), 1);
                end
                X0 = X0 +SpikeAFE.OversamplingRatio;
                X1 = X1 +SpikeAFE.OversamplingRatio;
                idx = idx +1;
            end
        end
        %% --- Digital Filtering
        function Xout = DigFilt(SpikeAFE, Xin, doFILT)
            if(doFILT)
                X0 = Xin - 2^(SpikeAFE.nBitADC-1);
                dX0 = SpikeAFE.aIIR_DIG* [X0 -SpikeAFE.MemIIR_DIG(1, 1:end)]';
                dX1 = SpikeAFE.bIIR_DIG* [dX0 SpikeAFE.MemIIR_DIG(1, 1:end)]';
                SpikeAFE.MemIIR_DIG(1,:) = [dX0 SpikeAFE.MemIIR_DIG(1, 1:end-1)];

                Xout = dX1 + 2^(SpikeAFE.nBitADC-1);
                SpikeAFE.XoldDIG = Xout;
            else
                Xout = SpikeAFE.XoldDIG;
            end
        end
        %% --- Threshold determination for neural input
        function Xout = Thres(SpikeAFE, Xin, mode)
            switch(mode)
                case 1 % Constant value
                    Xout = 0*Xin + SpikeAFE.ThrMin;
                case 2 % Standard derivation of background activity
                    Xout = 0*Xin + 8* mean(abs(Xin)/0.6745);
                case 3 % Automated calculation of threshold (using by BlackRock)
                    Xout = 0*Xin + 4.5* sqrt(sum(Xin.^2)/length(Xin));
                case 4 % Mean value 
                    Xout = 8* movmean(abs(Xin), [SpikeAFE.XWindowLength+SpikeAFE.InputDelay 0]);
                case 5 % Lossy peak detection
                    Xout = envelope(abs(Xin), 21, 'rms');                   
                case 6 % window mean method for max-detection
                    Xout = 0*Xin;
                    WindowLength = 20;
                    for i = 1:1:floor(length(Xin)/WindowLength)
                        X0 = [1 WindowLength] + (i-1)*WindowLength;
                        Xout(X0(1):X0(2)) = max(Xin(X0(1):X0(2))); 
                    end
                    Xout = 10*movmean(Xout, [200 0]); %mean(Xhi, 100);
                case 7
                    if isa(Xin, 'int16')
                        X0 = double(Xin);
                    else
                        X0 = Xin;
                    end
                    Xout = sgolayfilt(X0, 3, 31);
            end
            if isa(Xin, 'int16')
                Xout = int16(Xout);
            else
                Xout = double(Xout);
            end
        end
        %% --- Spike Detection       
        function [Xtrg, Xsda, Xthr] = SpikeDetection(SpikeAFE, Xin, mode, doSDA)
            % --- Einfache Variante für direkte Anwendung
            % Xsda = Xin(k+1:end-k).^2 - Xin(1:end-2*k).* Xin(2*k+1:end); 
            % [YPks, XPks] = findpeaks(Xsda, 'MinPeakHeight', Settings.ThresholdSDA, 'MinPeakDistance', round(500e-6*SampleRate)); 
            % --- Methoden
            % Auswahl erfolgt über die Vektor-Länge von dXsda 
            % length(x) == 1:   mit dX = 1 --> NEO, dX > 1 --> k-NEO
            % length(x) > 1:    M-TEO
            
            k = SpikeAFE.dXsda;           

            if(SpikeAFE.RealtimeMode)
                % Ausführung der Realtime SpikeDetection mit Speicher-Anpassung
                if(doSDA) 
                    SpikeAFE.MemSDA = [Xin SpikeAFE.MemSDA(1:end-1)];                        
                    % Bestimmung der NEO-Energie
                    if(~SpikeAFE.modeSDA)
                        Xsda = SpikeAFE.MemSDA(k+1).^2 - SpikeAFE.MemSDA(1)* SpikeAFE.MemSDA(end);
                    else
                        Xmteo = zeros(size(k));
                        for idx = 1:1:length(k)
                            Xmteo(idx) = SpikeAFE.MemSDA(k(end)+1).^2 - SpikeAFE.MemSDA(k(end)+1-k(idx))* SpikeAFE.MemSDA(k(end)+1+k(idx));
                        end
                        Xsda = max(Xmteo);
                        SpikeAFE.XsdaOld = Xsda;
                    end
                else
                    Xsda = SpikeAFE.XsdaOld;
                end
            else
                % Ausführung der Parallel SpikeDetection             
                if ~SpikeAFE.modeSDA
                    % --- Normale Ausführung
                    Xsda = Xin(k+1:end-k).^2 - Xin(1:end-2*k).* Xin(2*k+1:end);
                    if isa(Xin, 'int16')
                        mat = int16(ones(1,k));
                    else
                        mat = ones(1,k);
                    end
                    Xsda = [mat*Xsda(1) Xsda mat*Xsda(end)];
                else
                    % --- MTEO-Ausführung
                    if isa(Xin, 'int16')
                        Xmteo = int16(zeros(length(k), length(Xin)));
                    else
                        Xmteo = zeros(length(k), length(Xin));
                    end
                    for idx = 1:1:length(k)
                        kSDA = k(idx);

                        mat = ones(1,kSDA);
                        if isa(Xin, 'int16')
                            mat = int16(mat);
                        end
                        X0 = abs(Xin(kSDA+1:end-kSDA).^2 -Xin(1:end-2*kSDA).* Xin(2*kSDA+1:end));
                        Xmteo(idx,:) = [mat*X0(1) X0 mat*X0(end)];
                    end
                    Xsda = max(Xmteo);
                end            
                if(1)
                    figure;
                    plot(Xmteo')
                    hold on;
                    plot(Xsda)
                    legend({'k=1'; 'k=2'; 'k=3'; 'Xsda'})
                end
            end

            % --- Schwellenwert-Bestimmung
            Xthr = Thres(SpikeAFE, Xsda, mode);
            
            % --- Trigger-Generierung
            %Xtrg = logical(Xsda >= Xthr);
            if(mode == 7)
                [~, X] = findpeaks(single(Xthr), 'MinPeakWidth', 11, 'MinPeakDistance', SpikeAFE.XWindowLength, 'MinPeakHeight', mean(single(Xthr))+std(single(Xthr)));
                Xtrg = zeros(size(Xthr));
                Xtrg(X) = 1;
            else
                Xtrg = single(Xsda >= Xthr);
            end
        end        
        
        %% --- Frame Generation
        function [Frame, Xpos] = FrameGeneration(SpikeAFE, Xin, Xtrg)
            % --- Check if no results are available
            if(sum(Xtrg) == 0)
                Frame = [];
                Xpos = [];
                return;
            end
            % --- Extract x-positions from the trigger signal
            %Xpos = 1 + find(diff(Xtrg) == 1);
            
            [~, Xpos0] = findpeaks(Xtrg, 'MinPeakDistance', SpikeAFE.XWindowLength, 'MinPeakHeight', 0.7);
            % --- Extract frames
            if isa(Xin, 'int16')
                Frame = int16([]);
            else
                Frame = [];
            end
            Xpos = [];
            idx = 1;
            for idx = 1:1:length(Xpos0)
                dXneg = Xpos0(idx);
                dXpos = Xpos0(idx) +SpikeAFE.XWindowLength +SpikeAFE.XOffset -1;
                %dXneg = Xpos0(i) -SpikeAFE.XDeltaNeg -SpikeAFE.XOffset;
                %dXpos = Xpos0(i) +SpikeAFE.XDeltaPos +SpikeAFE.XOffset -1;
                if(dXneg >= 1 && dXpos <= length(Xtrg))    
                    Frame(idx,:) = Xin(dXneg:dXpos);   
                    Xpos(idx) = Xpos0(idx);
                    idx = idx +1;
                end            
            end
        end
        
        %% --- Frame Aligning
        function FrameOut = FrameAligning(SpikeAFE, FrameIn, AlignMode)
            if isa(FrameIn, 'int16')
                FrameOut = int16([]);
            else
                FrameOut = [];
            end
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
                
                Xpos0(x,:) = max_pos + [-SpikeAFE.XDeltaNeg 0 +SpikeAFE.XDeltaPos]; 
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
                        if isa(FrameIn, 'int16')
                            mat = int16(ones(1, abs(Xpos(3)-length(Frame0))-1));
                        else
                            mat = ones(1, abs(Xpos(3)-length(Frame0))-1);
                        end
                        FrameOut(x,:) = [Frame0(Xpos(1):end) Frame0(end)*mat];
                    case 2
                        if isa(FrameIn, 'int16')
                            mat = int16(ones(1, abs(Xpos(1))));
                        else
                            mat = ones(1, abs(Xpos(1)));
                        end
                        FrameOut(x,:) = [Frame0(1)*mat Frame0(1:Xpos(3))];                      
                end
            end
        end
        
        %% --- Function for Feature Extraction
        function [FeaturesArray, FeaturesCell] = FE_Normal(SpikeAFE, FrameIn)
          % Transfer from cell to array
          FrameArray = double(FrameIn)';
          
          FeaturesArray(1,:) = min(FrameArray');
          FeaturesArray(2,:) = max(FrameArray');
          FeaturesArray(3,:) = mean(FrameArray');
          for idx = 1:1:size(FrameArray,1) 
            FeaturesArray(4, idx) = length(find(FrameArray(idx,:) <= 0));
            FeaturesArray(5, idx) = length(find(FrameArray(idx,:) > 0));
          end
          FeaturesCell = SpikeAFE.ArrayToCell(FeaturesArray);

        end
        function [FeaturesArray, FeaturesCell] = FE_PCA(FeatExt, FrameIn)
          % Transfer from cell to array          
          [~, PCA_score] = pca(double(FrameIn));
          FeaturesArray = PCA_score(:,1:end)';
          FeaturesCell = FeatExt.ArrayToCell(FeaturesArray);

        end
        
        %% --- Function for Clustering
        function  Cluster = Clustering(SpikeAFE, FeatureTable)
            Cluster = transpose(kmeans(FeatureTable', SpikeAFE.NoCluster));
        end
        
        %% --- Function for determining spike ticks
        function SpikeTicks = DetermineSpikeTicks(FeatExt, XPos, ClusterIn, Time)
            SpikeTicks = zeros(FeatExt.NoCluster, length(Time));
            for idx = 1:1:FeatExt.NoCluster
                XCluster = find(ClusterIn == idx);
                SpikeTicks(idx, XPos(XCluster)) = 1;                    
            end
        end

        %% --- Function for analyzing accuracy of spike detection
        function results = analyzeSDA(SpikeAFE, Xin, Xchk, tol)    
            TP = 0;                 % number of true positive
            TN = 0;                 % number of true negative
            FP = 0;                 % number of false positive
            FN = 0;                 % number of false negative
            
            XposIn = find(single(Xin) == 1);
            for idxX = 1:length(XposIn)
                for idxY = 1:length(Xchk)
                    dX = XposIn(idxX) -Xchk(idxY);
        
                    % --- Decision tree
                    if(abs(dX) <= tol)
                        TP = TP + 1;
                        break;
                    elseif idxY == length(Xchk)
                        FP = FP + 1;
                        break;
                    end
                end
            end
            FN = length(Xchk) -TP -FP;
            TN = floor(length(Xin)/SpikeAFE.XWindowLength) - TP;
            
            % --- Output parameters
            results.FPR = FP/(FP +TN);      % False Positive rate - Probability of false alarm
            results.FNR = FN/(FN +TP);      % False Negative Rate - Miss rate
            results.TPR = TP/(TP +FN);      % True Positive rate - Sensitivity
            results.TNR = TN/(TN +FP);      % True Negative Rate - Specificity
            results.PPV = TP/(TP +FP);      % Positive predictive value
            results.NPV = TN/(TN +FN);      % Negative predictive value
               
            results.Accuracy    = (TP +TN)/(TP +TN +FN +FP);
        end    
    end
    % --- Functions only for internal
    methods(Access = private)
        %% --- Help function: A/D sampling
        function [Xout, Uout] = AD_Conv(SpikeAFE, Uin, EN)
            if(EN)
                % ADC output + Quantization noise
                NoiseQuant = randi(3)-2;
                Prep = find(Uin - SpikeAFE.partitionADC <= 0, 1);
                Xout = int16(Prep -2^(SpikeAFE.nBitADC-1) -1 + NoiseQuant);
                Uout = SpikeAFE.partitionADC(Prep) + (NoiseQuant - 1)*SpikeAFE.LSB/2;
                SpikeAFE.XoldADC = Xout;
                SpikeAFE.UoldADC = Uout;
            else 
                Xout = SpikeAFE.XoldADC;
                Uout = SpikeAFE.UoldADC;
            end
        end
        %% --- Help function: Transform cell to array
        function Array = CellToArray(SpikeAFE, CellIn)
            % --- alternative to cell2mat()
            ArraySize = [size(CellIn,2) size(CellIn{1},2)];
            Array = zeros(ArraySize);
            for i = 1:1:ArraySize(1)
                Array(i,:) = CellIn{i}; 
            end
        end        
        function Cell = ArrayToCell(SpikeAFE, ArrayIn)
            CellSize = size(ArrayIn);
            Cell = cell(1, CellSize(2));
            for i = 1:1:CellSize(2)
               Cell{i} = ArrayIn(:,i); 
            end
        end
    end    
end