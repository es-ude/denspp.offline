%% --- Settings (global)
% Realtime-Mode (0: offline, 1: online)
Settings.RealtimeMode = 0;
% Auswahl des zu verwendenen Datenquelle (0: Datensatz laden, 1: Alte Konfig.)
Settings.DataType = 1;
% Auswahl der Elektroden (= 0, ruft alle Daten auf)
Settings.CHsel = 0;
% Neuabtastungs-Rate der Eingangsdaten
Settings.DesiredFs = 100e3;
% Angabe des zu betrachteten Zeitfensters [Start Ende] in sec.
% (TRange = 0 ruft vollen Datensatz auf)
Settings.TRange = [10 40];
% Using ParallelComputing for MultiChannel-Auslese (0: disable, 1: enable)
Settings.EnableParallelComputing = 0;

%% --- Settings (Analogue Front-end)
% --- Supply voltage
AFE_SET.Udd = 0.6;      
AFE_SET.Uss = -0.6;
% --- Preamplification and bandpass filtering
AFE_SET.GainPre = 25;
AFE_SET.nFilt_ANA = 2;  
AFE_SET.fFilt_ANA = [200 5e3];
% --- Properties of ADC
AFE_SET.Oversampling = 1;
AFE_SET.nBitADC = 12;   
AFE_SET.dUref = 0.05;            
AFE_SET.SampleRate = 30e3;
% --- Digital filtering for ADC output and CIC
AFE_SET.nFilt_DIG = 2;  
AFE_SET.fFilt_DIG = [200 5e3];
% --- Properties of spike detection
AFE_SET.dXsda = [2 4 6];      
AFE_SET.ThresMode = 7;
AFE_SET.SDA_ThrMin = 1.5e-9; %(only for mode=1)
% --- Properties of Framing and Aligning of spike frames
AFE_SET.XOffset = round(0.5e-3* AFE_SET.SampleRate);
AFE_SET.XDeltaNeg = round(0.5e-3* AFE_SET.SampleRate);
AFE_SET.InputDelay = AFE_SET.XOffset;
AFE_SET.XWindowLength = round(2e-3* AFE_SET.SampleRate);
% --- Properties for Labeling data
AFE_SET.NoCluster = 19; 
