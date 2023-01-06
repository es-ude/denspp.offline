function plotResults_Transient(SignalsAFE, Fs, LSB, Labeling, noCH, doSHFT, Color, line)    
    Uin = SignalsAFE.Uin{noCH};
    Uadc = double(SignalsAFE.Xadc{noCH})* LSB;
    Xsda = double(SignalsAFE.Xsda{noCH});
    Xthr = SignalsAFE.Xthr{noCH};
    
    XPos = SignalsAFE.XPos{noCH};
    SpikeTicks = SignalsAFE.SpikeTicks{noCH};
    ClusterID = SignalsAFE.ClusterID{noCH};
    
    t = (0:1:length(Uin)-1) /Fs(1); 
    tADC = (0:1:length(Uadc)-1) /Fs(2); 

    % --- Labeling einfügen 
    if(Labeling.Exists)
        XLabel = tADC(Labeling.ADCXPosSpike);
        YLabel0 = 0.*XLabel + max(Uadc);
        YLabel1 = 0.*XLabel + max(Xsda);
    end
    XMeas = tADC(XPos);
    YMeas = 0*XMeas + max(Xsda); 

    h = figure;
    tiledlayout(3,1+doSHFT);
    
    %% --- Plot 1: Input and AFE
    ax(1) = nexttile;
    yyaxis left;
    stairs(t, 1e6*Uin, 'b', 'Linewidth', line);  
    ylabel('U_{in} (µV)');

    yyaxis right;
    stairs(tADC, 1e3*Uadc, 'r', 'Linewidth', line);
    grid on;    hold on;
    if(Labeling.Exists)
        plot(XLabel, 1e3*YLabel0, 'b', 'LineStyle', 'none', 'Marker', '*', 'Markersize', 4); 
    end

    setGraphicStyle(0);
    ylabel('U_{adc} (mV)');
    legend({'U_{in}'; 'U_{adc}'}, 'Location', 'NorthWest');
    
    % --- STFT of input signal
    if(doSHFT)
        ax = [ax nexttile];
        stft(Uin, 1/mean(diff(t)));
        ylim(10* [0 1]);
        setGraphicStyle(0);
    end

    % --- Spike Detection
    ax = [ax nexttile];
    stairs(tADC, Xsda, 'k', 'Linewidth', line);
    grid on; hold on;
    stairs(tADC, Xthr, 'r', 'Linewidth', line);
    plot(XMeas, 0.98*YMeas, 'k', 'LineStyle', 'none', 'Marker', '+', 'Markersize', 8); 
    if(Labeling.Exists)
        plot(XLabel, YLabel1, 'k', 'LineStyle', 'none', 'Marker', '*', 'Markersize', 4); 
    end
    setGraphicStyle(0);
    xlabel('Time t (s)');
    ylabel('U_{sda} (mV^2/s^4)');
    legend({'U_{SDA}'; 'U_{thr}'; 'Detected'; 'Labeling'}, 'Location', 'NorthWest');
        
    % --- STFT of Spike Detection
    if(doSHFT)
        ax = [ax nexttile];
        stft(Xsda, 1/mean(diff(tADC)));% 'r', 'Linewidth', line);
        ylim(10* [0 1]);    
        setGraphicStyle(0);
    end

    % --- SpikeTicks
    ax = [ax nexttile];
    NoCluster = size(unique(ClusterID, 'stable'),2);
    for i = 1:NoCluster
        plot(tADC, (i-1) + 0.8*SpikeTicks(i,:), 'Color', Color{i});
        if(i == 1)
            hold on; grid on;
        end
    end
    setGraphicStyle(0);
    set(gca, 'YTick', []);
    xlabel('Time t (s)');
    ylabel('Spike Ticks');
    
    set(gcf,'color','white')
    % --- Linking the x-axis of the plots
    linkaxes(ax, 'x');
    
    %% --- Plot 2: Frame Plots
    if(0)
        h = figure;
        tiledlayout(2,1);
        ax(1) = nexttile;
        hold on;    grid on;
        for i = 1:1:length(XPos)
           stairs(1e3*FramesOrig{i}); 
        end
        setGraphicStyle(0);
        ylabel('ADC output');
        xlabel('Position');
        
        ax(2) = nexttile;
        hold on;    grid on;
        for i = 1:1:length(XPos)
           stairs(1e3*FramesAlign{i}); 
        end
        setGraphicStyle(0);
        ylabel('ADC output');
        xlabel('Position');

        set(gcf,'color','white')
    end
end