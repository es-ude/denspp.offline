% Modell zur Ausführung des Hodgkin-Huxley-Modells
% https://de.mathworks.com/matlabcentral/fileexchange/69425-inside-the-brain-modeling-the-neuron?focused=4ed85dc3-6b98-4a96-a617-c11a84e536a8&tab=example
% Eingabe-Parameter (I = Stromverlauf [nA], dt = Zeitabstand [ms])

function [Vint0, m1, n1, h1] = modellMcNeal_0v1(Vext, Vint, T, dt, kT, m0, n0, h0, CFL) 
    Vint0 = transpose(Vint);
    Vext0 = transpose(Vext);
    %% Variablen
    % --- Gating-Variablen
    % Setting some parameters of the model (1), (2), (3)
    % Electrical Conductances of Sodium (Na), Potassium (K) and Leak Ions (L) [mS/cm2] 
    gNa = 120; 
    gK = 36;  
    gL = 0.3;  
    % Voltage Sources [mV] due to electrochemical gradients of Sodium (Na), Potassium (K) and Equivalent Ions (Eq)
    VNa = 55; 
    VK = -77; 
    VL = -54.4;
    V0 = -64.952;  
    
    % Capacity of the membrane [µF/cm2] 
    Cm = 4;
    % Temperature [K], propagation
    T0 = 279.45;    
    
    % Influence of temperature
    qM = 3* exp((T-T0)/10);
    qH = 3* exp((T-T0)/10); 
    qN = 3* exp((T-T0)/10);
    qT = mean([qM qH qN]);
    
    %% --- Berechnung der Activating functions
    % Thomas-Diagonal-Matrix erstellen
    k0 = -2* ones(1, length(Vext));
    k1 = ones(1, length(Vext)-1);
    A1 = diag(k0,0) + diag(k1,1) + diag(k1,-1);
    A1(1,1) = -1;        A1(1,2) = 1;
    A1(end,end) = -1;    A1(end,end-1) = 1;
    k2 = diag(ones(1, length(Vext)),0);
    
    method = 2;
    % Lösungsmatrix erstellen
    switch(method)
        case 0
            Aint = k2 + CFL* A1;
        case 1
            Aint = inv(k2 - CFL* A1);
        case 2
            Aint = (k2 + CFL/2*A1)* inv(k2 - 0.5*CFL*A1);
    end        
    Aext = CFL* A1;
    
    if(kT == 1)
        Vint0 = Vint0 + 1e-3*V0;
    end
    
    %% --- Berechnung
    VNode = 1e3* Vint0;
    alpha_m = 0.1.*(VNode + 40)./(1 - exp(-(VNode + 40)/10));
    beta_m = 4.*exp(-(VNode + 65)/18);
    alpha_h = 0.07.*exp(-(VNode + 65)/20);
    beta_h = 1./(1 + exp(-(VNode + 35)/10));
    alpha_n = 0.01.*(VNode + 55)./(1 - exp(-(VNode + 55)/10 ));
    beta_n = 0.125.*exp(-(VNode + 65)/80);
        
    if(kT == 1)
        m0 = alpha_m./ (alpha_m + beta_m);
        n0 = alpha_n./ (alpha_n + beta_n);
        h0 = alpha_h./ (alpha_h + beta_h);
    end

    iK = gK.* n0.^4.* (VNode -VK);
    iNa = gNa.* m0.^3.* h0.* (VNode -VNa);
    iL = gL.* (VNode -VL);

    dV = -1e3*(iK + iNa + iL)/Cm* dt;
    dm = alpha_m.*(1 - m0) - beta_m.* m0;
    dn = alpha_n.*(1 - n0) - beta_n.* n0;
    dh = alpha_h.*(1 - h0) - beta_h.* h0;

    fni = transpose((Aint)* transpose(VNode));
    fne = transpose((Aext)* transpose(1e3* Vext0));
    
    Vint0 = fni + fne + dV* qT;
    m1 = m0 + dm* qM* 1e3* dt;
    n1 = n0 + dn* qN* 1e3* dt;
    h1 = h0 + dh* qH* 1e3* dt;
    
    m1(find(m1 >= 1)) = 1; m1(find(m1 <= 0)) = 0; 
    n1(find(n1 >= 1)) = 1; n1(find(n1 <= 0)) = 0; 
    h1(find(h1 >= 1)) = 1; h1(find(h1 <= 0)) = 0; 
end