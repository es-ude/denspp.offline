function plotResults_Threshold(SignalsAFE, Fs, LSB, Color, line)
    FontSize = 16;
    noCH = 1;
    tZoom = [11.78 11.92];
    
    % --- Preselection for cut-out
    zIn = length(SignalsAFE.Uin{noCH});
    zAdc = length(SignalsAFE.Xadc{noCH});
    t = (0:1:zIn-1) /Fs(1);
    tADC = (0:1:zAdc-1) /Fs(2); 

    X00 = find(t >= tZoom(1), 1);
    X01 = find(t >= tZoom(2), 1);
    X10 = find(tADC >= tZoom(1), 1);
    X11 = find(tADC >= tZoom(2), 1);

    % --- Final selection
    Uin = SignalsAFE.Uin{noCH}(X00:X01);
    Uadc = double(SignalsAFE.Xadc{noCH}(X10:X11))* LSB;
    Xsda = double(SignalsAFE.Xsda{noCH}(X10:X11));
    Xthr = SignalsAFE.Xthr{noCH}(X10:X11);
    t = (0:1:length(Uin)-1) /Fs(1);
    tADC = (0:1:length(Uadc)-1) /Fs(2);
    
    XPos = SignalsAFE.XPos{noCH};
    Y0 = find(XPos >= X10, 1);
    Y1 = find(XPos >= X11, 1);

    Frames = SignalsAFE.FramesOrig{noCH}(Y0:Y1, :);
   
    % --- Threshold
    Yth0 = 4*mean(abs(Uin)/0.6745);    
    Yth1 = threshold(Uin, 501);

    Yth2 = 4*mean(abs(Xsda)/0.6745); 
    Yth3 = threshold(Xsda, 201);
       
    % --- Plotten
    h = figure;
    set(gcf, 'color', 'w');
    set(gcf, 'units', 'points', 'Position', [900, 50, 800, 450]);
    tiledlayout(2, 3);
    
    ax(1) = nexttile(1, [1 2]);
    plot(t, 1e6* Uin, 'k', 'LineWidth', line);
    grid on;    hold on;
    plot(t, 0* Uin + 1e6* Yth0, 'g', 'LineWidth', line);
    plot(t, 0* Uin + 1e6* Yth1, 'r', 'LineWidth', line);
    plot(t, 0* Uin - 1e6* Yth0, 'g', 'LineWidth', line);
    plot(t, 0* Uin - 1e6* Yth1, 'r', 'LineWidth', line);

    ylabel('U_{in} (µV)');
    set(gca, 'FontName', 'Times', 'FontSize', FontSize);
    legend({"U_{in}"; "U_{th0}"; "U_{th1}"});
    
    ax(2) = nexttile(4, [1 2]);
    plot(tADC, Xsda, 'k', 'LineWidth', line);
    grid on; hold on;
    plot(tADC, 0* Xsda + Yth2, 'g', 'LineWidth', line);
    plot(tADC, 0* Xsda - Yth3, 'r', 'LineWidth', line);
    
    xlabel('Time t (s)');
    ylabel('X_{neo} (V/cm^2)');
    set(gca, 'FontName', 'Times', 'FontSize', FontSize);
    linkaxes(ax, 'x');
    legend({"X_{neo}"; "X_{th0}"; "X_{th1}"});

    nexttile(3, [2 1]);
    stairs(Frames', 'LineWidth', line);
    grid on;

    xlabel('Frame position')
    ylabel('U_{sp} (µV)')
    set(gca, 'FontName', 'Times', 'FontSize', FontSize);
end


function Xout = threshold(Xin, Mlength)
    for(i = 1:length(Xin))
        X0 = i - floor(Mlength/2);
        X1 = i + floor(Mlength/2);
        if(X0 < 1)
            X0 = 1;
            Mlength0 = i; 
        end
        if(length(Xin) < X1)
            X1 = length(Xin);
            Mlength0 = length(Xin) - i;
        end
        Mlength0 = X1 - X0 +1;
        Yth1(i) = sumsqr(Xin(X0:X1))/Mlength0;
    end
    Xout = -2.5* sqrt(Yth1);
end