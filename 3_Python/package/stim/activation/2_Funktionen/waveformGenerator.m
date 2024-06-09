%% --- Header
    %Author:            A. Erbslöh
    %Erstelldatum:      30.05.2015
    %Aktualisierung:    27.05.2020
    %Company:           EBS
    %Version:           1v1
    %Projekt:           Modellierung einer selbstregelnden Motors
    %Bemerkungen:       /

    %% --- Funktion
function [sigwave, stilWFG] = waveformGenerator(sig_form, a, lgth_vector)
%% --- Bestimmung der kathodischen Stimulationshälfte
    Fstim = zeros(1, lgth_vector);
    xStart = 1;
    xEnde = lgth_vector;
    stilWFG = '';
    for i= xStart:1:xEnde
       switch(sig_form)
           case 1   % Rechteckig
               Fstim(i) = a;
               stilWFG = 'Rect';
           case 2   % Linear steigend
               Fstim(i) = a*(i-xStart)/lgth_vector;
               stilWFG = 'LinInc';
           case 3   % Linear fallend
               Fstim(i) = a*(xEnde - i)/lgth_vector;
               stilWFG = 'LinDec';
           case 4   % Sägezahn
               Fstim(i) = a* (1 - abs(sawtooth(2*pi*(i-xStart)/lgth_vector)));
               stilWFG = 'Sawtooth';
           case 5   % Sinus
               Fstim(i) = a*sin(pi*(i-xStart)/lgth_vector);
               stilWFG = 'Sinus';
           case 6   % Sinus invertiert
               Fstim(i) = a*(1 - sin(pi*(i-xStart)/lgth_vector));
               stilWFG = 'SinInv';
           case 7   % Gaußförmig
               Fstim(i) = a*exp(-21/lgth_vector.^2.*(i-round(lgth_vector/2)).^2);
               stilWFG = 'Gauß';
       end                       
    end  
    sigwave = Fstim;
end
    
