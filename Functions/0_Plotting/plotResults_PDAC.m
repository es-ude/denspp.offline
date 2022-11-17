function plotResults_PDAC(Signals, Quality_Param, noCH, Color, line)
    FramesOrig = Signals.FramesOrig{noCH};
    FramesAlign = Signals.FramesAlign{noCH};
    ClusterID = Signals.ClusterID{noCH};
   
    NoCluster = size(unique(ClusterID, 'stable'),2);
    % --- Separating different spike waveforms from clustering
    SpikeMean = cell(1, NoCluster);
    
    SpikeArray = FramesAlign;
    for i = 1:NoCluster
        XCluster = find(ClusterID == i);
        YCluster = double(SpikeArray(XCluster,:));
        
        SpikeMean{i} = mean(YCluster);
        [~, Xmax{i}] = max(SpikeMean{i});
        [~, Xmin{i}] = min(SpikeMean{i});
    end
    Xlength = length(SpikeMean{1});
    
    %% --- Plot 1: Frames Input
    figure;
    set(gcf, 'color', 'w');
    set(gcf, 'units', 'points', 'Position', [900, 50, 800, 350]);

    tiledlayout(1, NoCluster);
       
    %--- Extracted Spike-Waveformes
    
    Scale = 1;
    for i = 1:NoCluster
        ax(i) = nexttile;  
        % --- PDAC
        A = 1:Xmin{i};
        B = Xmin{i}:Xlength;
        Baseline = SpikeMean{i}(Xmin{i});

        a0 = area(A, SpikeMean{i}(A), Baseline);      
        a0.FaceAlpha = 0.5;
        hold on; grid on;
        a0 = area(B, SpikeMean{i}(B), Baseline);
        a0.FaceAlpha = 0.5;

        % --- Mean waveform
        stairs(Scale* SpikeMean{i}, 'k', 'Linewidth', 2*line);
        
        setGraphicStyle(0);
        if(i == 1)
            ylabel('U_{in} (µV)')
        end        
        xticks([1 20 40 60]);
        xlim([1 size(SpikeArray,2)]);
        xlabel('Frame Position'); 
    end    
    
     
    
end