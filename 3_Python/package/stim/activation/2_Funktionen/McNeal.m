classdef McNeal < handle
   
    properties(GetAccess = public, SetAccess = private)
       
    end
   
    properties(Access = private)
        noNode = 0;
        lengthT = 0;
        % --- 
        % --- Modellparameter
        % Electrical Conductances of Sodium (Na), Potassium (K) and Leak Ions (L) [mS/cm2] 
        gNa = 120;
        gK = 36;  
        gL = 0.3;  
        % Voltage Sources [mV] due to electrochemical gradients of Sodium (Na), Potassium (K) and Equivalent Ions (Eq)
        VNa = 55; 
        VK = -77; 
        VL = -54.4;
        V0 = -64.952;
        Cm = 4;                 % Capacity of the membrane [µF/cm2] 
        T0 = 279.45;            % Temperature [K], propagation
        qM = 3* exp((T-T0)/10); % Influence of temperature
        qH = 3* exp((T-T0)/10); 
        qN = 3* exp((T-T0)/10);
        qT = mean([qM qH qN]);
    end
   
    methods(Access = public)
        function Init(noNode, lengthT)
            McNeal.noNode = noNode;
            McNeal.lengthT = lengthT;
        end
    end
   
    methods(Access = private)
       
    end
    
end