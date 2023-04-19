close all;  clear all;  clc;

%% --- Settings
setOptions.do_plot_check1 = true;
setOptions.do_plot_check2 = true;
setOptions.do_plot_result = true;

setOptions.do_save = true;
setOptions.do_2nd_run = false;
addon = "_Sorted";

% Settings für Martinez
criterion_Check = [3 0.7];
criterion_Run0 = 0.98;
criterion_Run1 = [0.05 0.025];

%% --- Vorverarbeitung: Datenaufnahme
% --- Daten laden
[filename, path2data] = uigetfile('../src_ai/data/*.mat');

% --- Multiselect-Auswertung
setOptions.path2file = strcat(path2data, filename);
setOptions.path2save = strcat(path2data, filename(1:end-4), addon, ".mat");
setOptions.path2fig = strcat(path2data, filename(1:end-4));

load(setOptions.path2file);
clear filename addon path2data;

% --- Bestimmung der Frame-Positionen und Anzahl der Cluster
IteNo = 1;
input_cluster_xpos = cell(1, length(unique(frames_cluster)));
for idx = min(frames_cluster):1:max(frames_cluster)
    input_cluster_xpos{IteNo} = find(frames_cluster == idx);
    IteNo = IteNo +1;
end
clear IteNo idx;

%% --- Vorverarbeitung: Konsistenz-Prüfung der Frames pro Cluster
frames_check_later = cell(1, size(input_cluster_xpos, 2));
process_frames_pos = cell(1, size(input_cluster_xpos, 2));

% Korrelation, ob Mean-Waveform vom Cluster mit Frames übereinstimmt
noRuns = size(input_cluster_xpos, 2);
num_frames_used = 0;
num_frames_missed = 0;
for idx = 1:1:noRuns
    do_run = true;
    IteNo = 0;
    XCheck = input_cluster_xpos{idx};
    XCheck_False = [];
    check = [];
    while(do_run)
        % Falsche rausfiltern
        XCheck(check) = [];
        Ycheck = double(frames_in(XCheck,:));
        Ymean = mean(Ycheck);
        WaveRef = crossval(Ymean, Ymean);

        metric_Check = zeros(length(XCheck), 6);
        for idy = 1:1:size(Ycheck, 1)
            WaveIn = crossval(Ycheck(idy,:), Ymean);     
            metric_Check(idy, :) = [calc_metric(WaveIn, WaveRef), XCheck(idy)];
        end
    
        % Kritierien fürs Rausfiltern
        check = find((abs(metric_Check(:,1)) > criterion_Check(1)) | (metric_Check(:,5) < criterion_Check(2)));
        XCheck_False = [XCheck_False; metric_Check(check,6)];
        if(~isempty(check) && (IteNo <= 3))
            do_run = true;
        else
            do_run = false;
        end        
        
        IteNo = IteNo +1;
    end
    
    % Übergabe
    process_frames_pos{idx} = XCheck;
    num_frames_used = num_frames_used + length(XCheck);
    frames_check_later{idx} = XCheck_False; 
    num_frames_missed = num_frames_missed + length(XCheck_False);     
end
clear diffX A do_run check Ycheck XCheck XCheck_False Ymean WaveIn WaveRef;
clear noRuns idx idy IteNo;
clear metric_Check;

%% --- Cluster mergen 
new_frames_cluster = int16([]);
new_frames_pos = {};
XCheck_False = {};

SelCluster = 1;
IteMiss = 1;
IteNo = 1;
for idx = 1:size(process_frames_pos, 2)
    Xin = process_frames_pos{idx};
    Yin = double(frames_in(Xin,:));

    if(idx == 1)
        %--- Initialisierung
        Xout = Xin;
        SelCluster = 1;    
    else
        %--- Erste Prüfung: Mean-Waveform vergleichen mit bereits gemergten Clustern
        metric_Run0 = [];
        Ycheck = {};
        for idy = 1:1:size(new_frames_pos, 2)
            % --- Prüfung der Mean-Formen
            Xcheck = new_frames_pos{2, idy};
            Ycheck{idy} = double(frames_in(Xcheck,:));
            Ymean = mean(Ycheck{idy});
            
            WaveRef = crossval(Ymean, Ymean);
            WaveIn = crossval(mean(Yin), Ymean);     
            metric_Run0(idy, :) = calc_metric(WaveIn, WaveRef);
        end
    
        % --- Auffinden von Abweichlern
        do_rerun = true;
        OutX = [];
        while(do_rerun)
            do_rerun = false;
        end    
        clear idy Xcheck Ymean WaveRef Wavein

        % --- Entscheidung treffen
        if setOptions.do_plot_check1
            close all;
            figure();
            color = 'rbyg';   
            plot(Yin', 'k');
            hold on;    grid on;
            for idy = size(Ycheck)
                plot(Ycheck{idy}', color(1+mod(idy,4)))
            end
        end
        [candY, candX] = max(metric_Run0(:, 5));
        if(isempty(candX))
            % Keine Lösung vorhanden --> Anhängen
            mode = 0;
        else 
            if(candY >= criterion_Run0)
                % Match
                mode = 2;  
            else
                % Neues Cluster
                mode = 1;
            end
        end

        % --- Ausführung
        switch(mode)
            case 0
                % --- Entscheidung: Sammeln und am Ende prüfen
                SelCluster = 0;
                Xout = Xin;
            case 1
                % --- Entscheidung: Neues Cluster
                SelCluster = max(new_frames_cluster) +1;
                Xout = Xin;
            case 2
                % --- Entscheidung Bedingung erfüllt   
                SelCluster = candX;
                Xold = Xin;
                Xnew = new_frames_pos{2, SelCluster};
                Xnew(OutX) = [];
                Xcombine = [Xin; Xnew];
    
                % Zweite Prüfung (opt): Einzel-Waveform prüfen    
                if(setOptions.do_2nd_run)
                    metric_Run1 = [];
                    metric_Run2 = [];
                    % Erste Correlation: Einzel (alt) mit Mean-Waveform
                    Ycheck0 = double(frames_in(Xin, :));
                    Ymean0 = mean(Ycheck0);
                    WaveRef = crossval(Ymean0, Ymean0);
                    for idz = 1:1:size(Ycheck0,1)
                        WaveIn = crossval(Ycheck0(idz,:), Ymean0);
                        metric_Run1(idz, :) = calc_metric(WaveIn, WaveRef);
                    end
    
                    % Zweite Correlation: Einzel (Neu) mit Mean-Waveform
                    Ycheck1 = double(frames_in(Xnew, :));
                    Ymean1 = mean(Ycheck1);
                    WaveRef = crossval(Ymean1, Ymean1);
                    for idz = 1:1:size(Ycheck1,1)
                        WaveIn = crossval(Ycheck1(idz,:), Ymean1);
                        metric_Run2(idz, :) = calc_metric(WaveIn, WaveRef);
                    end
                    
                    % Dritte Correlation: Einzel (kombiniert) mit Mean-Waveform
                    Ycheck2 = double(frames_in(Xcombine,:));
                    Ymean2 = mean(Ycheck2);
                    WaveRef = crossval(Ymean2, Ymean2);
                    for idz = 1:1:size(Ycheck2,1)
                        WaveIn = crossval(Ycheck2(idz,:), Ymean2);
                        metric_Run3(idz, :) = calc_metric(WaveIn, WaveRef);
                    end
                    
                    % Parameter extrahieren und entscheiden
                    check.input_mean = mean(metric_Run1(:,5));
                    check.input_std = std(metric_Run1(:,5));
                    check.new_mean = mean(metric_Run2(:,5));
                    check.new_std = std(metric_Run2(:,5));
                    check.output_mean = mean(metric_Run3(:,5));
                    check.output_std = std(metric_Run3(:,5));
                    check.diff_mean = abs(check.input_mean - check.output_mean);
                    check.diff_std = abs(check.input_std - check.output_std);


                    check.result = (check.diff_mean <= criterion_Run1(1)) && check.diff_std <= criterion_Run1(2);
    
                    % Check point mit Variable check
                    if setOptions.do_plot_check2
                        figure(1);
                        tiledlayout(3,1);
                        nexttile;
                        plot(Ycheck0', 'k'); 
                        grid on; hold on;
                        plot(Ymean0, 'r');
                        ylabel("X_{in}");
                        title(strcat("Cor = ", num2str(check.input_mean, '%.4f'), " - Std = ", num2str(check.input_std, '%.4f')))
                        Ylimit = ylim;
    
                        nexttile;
                        plot(Ycheck1', 'k'); 
                        grid on; hold on;
                        plot(Ymean1, 'r');
                        ylabel("X_{new}");
                        title(strcat("Cor = ", num2str(check.new_mean, '%.4f'), " - Std = ", num2str(check.new_std, '%.4f')))
                        ylim(Ylimit);
    
                        nexttile;
                        plot(Ycheck2', 'k'); 
                        grid on; hold on; 
                        plot(Ymean2, 'r');
                        ylabel("X_{combine}")
                        title(strcat("Cor = ", num2str(check.output_mean, '%.4f'), " - Std = ", num2str(check.output_std, '%.4f')))
                        subtitle(strcat("Mode: ", num2str(check.result), " - \Delta mean = ", num2str(check.diff_mean, '%.4f'), " - \Delta std = ", num2str(check.diff_std, '%.4f')));
                        ylim(Ylimit);
    
                        pause(2);
                    end
                    clear Ycheck0 Ymean0 Ycheck1 Ymean1 Ycheck2 Ymean2 WaveIn WaveRef
                    clear OutX do_rerun;
                else
                    check.result = true;
                end

                % Ausführen
                if(check.result)
                    %--- Gute Übereinstimmung -> Merge
                    Xout = Xcombine;
                    SelCluster = unique(candX);
                else
                    %--- Schlechte Übereinstimmung -> Neu
                    Xout = Xin;
                    SelCluster = max(new_frames_cluster) +1;   
                end  
        end   
    end    
    % --- Übergabe
    new_frames_cluster(Xout) = SelCluster;
    if(SelCluster == 0)
        XCheck_False{IteMiss} = Xout;
        IteMiss = IteMiss +1;
    else
        new_frames_pos{1, SelCluster} = SelCluster;
        new_frames_pos{2, SelCluster} = Xout;
    end
end
clear idx idy idz IteNo Xcheck Ycheck Xout;
clear check Ycheck0 Ycheck1 Ymean0 Ymean1 WaveRef WaveIn;
clear candX candY SelCluster mode;

%% --- Neuen Output generieren und SNR berechnen
snr_mean = [];
snr_std = [];

for idx = 1:size(new_frames_pos, 2)
    selx = new_frames_pos{2,idx};
    new_frames_pos{3, idx} = frames_in(selx, :);

    Yin = double(new_frames_pos{3, idx});
    val_snr = zeros(1, size(Yin, 1));
    for idy = 1:length(Xin)
        val_snr(idy) = calculate_snr(Yin(idy,:), mean(Yin));
    end
    snr_mean(idx) = mean(val_snr);
    snr_std(idx) = std(val_snr);

end
clear idx idy  Xin Yin Ymean val_snr;

%% --- Plot results
close all;

if(setOptions.do_plot_result)
    sizeID = size(new_frames_pos, 2);
    if(sizeID >= 9)
            SubFig = [3, 3];
    else
        if(sizeID <= 4)
            SubFig = [2, 2];
        else
            SubFig = [3, 3];
        end
    end
    noSubFig = SubFig(1)* SubFig(2);
    for idy = 1:1:ceil(sizeID/noSubFig)
        figure('doublebuffer', 'off', 'visible', 'off');
        set(gcf, 'units','normalized','outerposition',[0 0 0.75 1]);
            
        selID = (1:1:noSubFig) + noSubFig*(idy-1);
        if(selID(end) > sizeID)
            selID = selID(1):1:sizeID;
        end
    
        IteNo = 1;
        for idx = 1:1:length(selID)
            NoCluster = selID(idx);
            % Decision, if more than 100 frames
            Yin = double(new_frames_pos{3, NoCluster});
            if(size(Yin,1) > 2000)
                selFrames = randperm(size(Yin,1), 2000);
            else
                selFrames = 1:size(Yin,1);
            end
            
            % Plot
            subplot(SubFig(1), SubFig(2), IteNo);
            IteNo = IteNo +1;
            
            plot(Yin(selFrames,:)', 'b');
            hold on; grid on;
            plot(mean(Yin), 'r', 'Linewidth', 1);
    
            t = title(strcat("Cluster ID: ",num2str(NoCluster), " - SNR = ", num2str(snr_mean(NoCluster)), "(\pm", num2str(snr_std(NoCluster)), ") dB - No = ", num2str(size(Yin,1))));
            set(gca, 'FontSize', 12);
            t.FontSize = 10;
        end        
        saveas(gcf, strcat(setOptions.path2fig, '_Results_Fig', num2str(idy, '%02d'), '.jpg'));
    end
    clear Yin Xout
    clear t idx idy SubFig IteNo selID noSubFig sizeID NoCluster;
end

%% --- Transfer to new file
output.frames = [];
output.cluster = [];
for idx = 1:size(new_frames_pos, 2)
    X = new_frames_pos{1, idx} .* ones(size(new_frames_pos{2, idx}, 1), 1);
    Z = new_frames_pos{3, idx};
    if idx == 1
        output.cluster = X;
        output.frames = Z;
    else
        output.cluster = [output.cluster; X];
        output.frames = [output.frames; Z];
    end
end


frames_in = output.frames;
frames_cluster = int16(output.cluster-1);
save(setOptions.path2save, 'frames_in', 'frames_cluster');
clear FileName;

%% --- Test plot
test_plot(frames_in, frames_cluster, 1);

disp("Ende");

%% --- External function
function test_plot(frames_in, cluster, idx)
    
    Xsel = find(cluster == idx);
    frames_sel = frames_in(Xsel, :);
    mean0 = mean(frames_sel);

    figure();
    plot(frames_sel', 'k')
    hold on;
    plot(mean0, 'r')
end

function out = calc_metric(WaveIn, WaveRef)
    [Y1, X1] = max(WaveIn);
    [Y2, X2] = max(WaveRef);
    
    out = [];
    % --- Normal results
    out(1) = X1 - X2;
    out(2) = sum(abs(WaveIn - WaveRef).^2);
    out(3) = trapz(WaveIn(1:X1)) - trapz(WaveIn(X1:end));
    out(4) = X1;
    out(5) = Y1;
end

function out = shuffle(in)
     out = in(randperm(length(in)));
 end

function out = crossval(Wave1, Wave2)
    out = xcorr(Wave1, Wave2)/(sqrt(sum(Wave1.^2))* sqrt(sum(Wave2.^2)));
end

function outdB = calculate_snr(waveIn, waveOut)    
    A = sum(waveIn.^2);
    B = sum((waveOut - waveIn).^2);
    outdB = 10*log10(A/B); 
end

function checkData(WaveIn, noID)
    WaveRef = mean(WaveIn);
    Yref = crossval(WaveRef, WaveRef);
    metric_test = zeros(size(WaveIn, 1), 5);
           
    for idz = 1:1:size(WaveIn, 1)
        Yin = crossval(WaveIn(idz,:), WaveRef);
        metric_test(idz, :) = calc_metric(Yin, Yref);
    end
    
    str = strcat('Cluster-ID: ', num2str(noID), ' - Crossval, Mean = ', num2str(mean(metric_test(:,5))), ' +/- ', num2str(std(metric_test(:,5))));
    disp(str);

    figure();
    subplot(2,1,1);
    p1 = plot(WaveIn', 'b', 'Linewidth', 1);       
    hold on;                grid on;
    p2 = plot(WaveRef, 'r', 'Linewidth', 1);  
    title(str);

    subplot(2,1,2);
    p3 = histogram(metric_test(:,5),100);

    % --- Warte-Routine
    waitforbuttonpress

    close gcf;    
end