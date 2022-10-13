function plotResults_FeatureMap(Signals, Quality_Param, noCH, Color, line)
    XPos = Signals.XPos{noCH};
    FramesOrig = Signals.FramesOrig{noCH};
    FramesAlign = Signals.FramesAlign{noCH};
    Features = Signals.Features{noCH};
    ClusterID = Signals.ClusterID{noCH};
   
    NoCluster = size(unique(ClusterID, 'stable'),2);
    % --- Separating different spike waveforms from clustering
    SpikeMean = cell(1, NoCluster);
    SpikeUp = cell(1, NoCluster);
    SpikeDown = cell(1, NoCluster);
    
    SpikeArray = FramesAlign;
    for i = 1:NoCluster
        XCluster = find(ClusterID == i);
        YCluster = double(SpikeArray(XCluster,:));

        SpikeMean{i} = mean(YCluster);
        SpikeUp{i} = max(YCluster);
        SpikeDown{i} = min(YCluster);
    end
    
    %% --- Plot 1: Frames Input
    figure;
    tiledlayout(2,2);
    if isa(FramesOrig, 'int16')
        Scale = 1;
        ytext{1} = 'ADC code';
    else
        Scale = 1e3;
        ytext{1} = 'ADC output (mV)';
    end
    
    % --- Original Spike Frames
    ax(1) = nexttile;
    stairs(Scale* FramesOrig'); 
    xlim([1 size(FramesOrig,2)]);
    hold on;    grid on;
    
    setGraphicStyle(0);
    xlabel('Frame Position');   
    ylabel(ytext);
   
    %--- FeatureMap
    ax(3) = nexttile;
    for i = 1:NoCluster
        XCluster = find(ClusterID == i);
        scatter3(Scale* Features(1,XCluster), Scale* Features(2,XCluster), Scale* Features(3,XCluster), Color{i}, 'filled');
        %scatterhist(Features(1,XCluster), Features(2,XCluster), 'color', Color{i}, 'Kernel', 'on');
        if(i == 1)
            hold on;    grid on;
        end
    end
    view([0 90]);
    setGraphicStyle(0);
    xlabel('Feature A');
    ylabel('Feature B');
    zlabel('Feature C');

    % Adding text to the plot
    if(~isempty(Quality_Param))
        TextXY(1) = 0.9* min(xlim);
        TextXY(2) = 0.8* min(ylim);
        Text{1} = ['Detect. rate = ', num2str(round(100* Quality_Param.DR{noCH}, 2)), ' %'];
        Text{2} = ['Comp. ratio = ', num2str(round(Quality_Param.CR{noCH},1))];
        text(TextXY(1), TextXY(2), Text);
    end
    
    % --- Aligned Spike Frames
    ax(2) = nexttile;
    stairs(Scale* FramesAlign'); 
    xlim([1 size(FramesAlign,2)]);
    hold on;    grid on;
    
    setGraphicStyle(0);
    xlabel('Frame Position');   
    ylabel(ytext);
    
    %--- Extracted Spike-Waveformes
    ax(4) = nexttile;           
    for i = 1:NoCluster
        inBetweenRegionX = [1:length(SpikeUp{i}), length(SpikeDown{i}):-1:1];
        inBetweenRegionY = [SpikeUp{i}, fliplr(SpikeDown{i})];
        fill(inBetweenRegionX, Scale* inBetweenRegionY', Color{i}, 'FaceAlpha', 0.1);
        if(i == 1)
            hold on; grid on;
        end        
        plot(Scale* SpikeMean{i}, 'Color', Color{i}, 'Linewidth', 2*line);
        plot(Scale* SpikeUp{i},   'Color', Color{i}, 'Linewidth', 0.25*line)
        plot(Scale* SpikeDown{i}, 'Color', Color{i}, 'Linewidth', 0.25*line);
    end    
    
    setGraphicStyle(0);
    xlim([1 size(SpikeArray,2)]);
    xlabel('Frame Position');
    ylabel(ytext);

    set(gcf,'color','white');
    
    %% --- Plot 2: Spike frames separately with Histo
    if(0)
        for i = 1:NoCluster
           f(i) = figure;
           X = 1:1:length(SpikeMean{i});
           h1 = scatterhist(X, SpikeMean{i}, 'Marker', '.', 'MarkerSize', 0.1, 'Color', Color{i}, 'NBins',[15,15]);
           xlabel('Time(s)');
           ylabel('Distance (m)');
           h1(1).Position([2 4]) = [0.13 h1(1).Position(4)+h1(1).Position(2)-0.13];
           h1(3).Position([2 4]) = h1(1).Position([2 4]);
           delete(h1(2));

           hold on; grid on;       
           inBetweenRegionX = [1:length(SpikeUp{i}), length(SpikeDown{i}):-1:1];
           inBetweenRegionY = [SpikeUp{i}, fliplr(SpikeDown{i})];
           fill(inBetweenRegionX, inBetweenRegionY', [0.75 0.75 0.75], 'FaceAlpha', 0.5);
           plot(SpikeMean{i}, 'Color', Color{i}, 'Linewidth', 2*line);
           plot(SpikeUp{i}, 'Color', Color{i}, 'Linewidth', 0.5*line)
           plot(SpikeDown{i}, 'Color', Color{i}, 'Linewidth', 0.5*line);

           PlotPos{i} = [0, 0, 0.5, 1];

           set(gcf,'color','white')
        end
    end 
end