  int input = 0; short signed in;
  short signed delay0 = 0; short signed delay1 = 0; short signed delay2 = 0; 
  short signed coeff_a1 = 1300; float coeff_a2 = 9600; 
  long signed A_1; long signed A_2; long signed S0; long signed S1; long signed NEO;
  short signed a1; short signed a2;
  short signed out;

  bool maximum; bool minimum;
  int max_cnt = 0; int min_cnt = 0; long time = 0; double t = 0; int norm_in = 0;

  long signed out_calc; long signed delay0_calc;

  int electrode = A0; int TRIGG_MAX = 13; int TRIGG_MIN = 8; int max = 0; int min = 0;

void setup() {
  Serial.begin(115200);
  pinMode(TRIGG_MAX, OUTPUT); pinMode(TRIGG_MIN, OUTPUT);
  analogReference(INTERNAL);
  digitalWrite(TRIGG_MAX,LOW);
  digitalWrite(TRIGG_MIN,LOW);
}

void loop() {
  t = t + 0.03;     // Konstante = Frequenz in Hertz/100
 // input = analogRead(electrode);
   input = 100 * (sin(t) + 1); // Testbench für adaptive Amplitudenanpassung

  if(input > norm_in){
    norm_in = input;
  }

  input = input * (1024/norm_in);
  // in = 10000*sin(t);   // Testbench für Sinus Rohdaten
  in = (input - 512)*64;
  
  A_1 = delay0 * coeff_a1; a1 = A_1/32767;
  A_2 = delay1 * coeff_a2; a2 = A_2/32767;
  out = in + a1 + a2;
  out_calc = out; delay0_calc = delay0;
  S0 = out_calc * out_calc; S1 = out_calc * delay0_calc; NEO = S0 - S1;
  delay2 = delay1;
  delay1 = delay0;
  delay0 = out;


  if(NEO > 1073741823 && in > 0){
    maximum = 1;
    minimum = 0;
    max_cnt = max_cnt + 1;
    min_cnt = 0;
    time = millis();
  }
  if(NEO > 1073741823 && in < 0){
    maximum = 0;
    minimum = 1;
    min_cnt = min_cnt + 1;
    max_cnt = 0;
    time = millis();
  }
  else{
    maximum = 0;
    minimum = 0;
  }

  if(millis() - time >= 300){
    max_cnt = 0;
    min_cnt = 0;
    digitalWrite(TRIGG_MAX,LOW);
    digitalWrite(TRIGG_MIN,LOW);
    max = 0; min = 0;
  }
  if(max_cnt >= 5){
    digitalWrite(TRIGG_MAX,HIGH);
    digitalWrite(TRIGG_MIN,LOW);
    max = 30000;
    min = 0;
  }
  if(min_cnt >= 6){
    digitalWrite(TRIGG_MAX,LOW);
    digitalWrite(TRIGG_MIN,HIGH);
    min = 30000;
    max = 0;
  }
  if((max_cnt == 0) && (min_cnt == 0)){
    digitalWrite(TRIGG_MAX,LOW);
    digitalWrite(TRIGG_MIN,LOW);
    max = 0; min = 0;
  }
  Serial.print(in); Serial.print(" "); Serial.print(max); Serial.print(" "); Serial.print(min); Serial.print(" "); Serial.print(max_cnt); Serial.print(" "); Serial.println(min_cnt);
  //if(max_cnt != 4 and min_cnt != 4){
  //  digitalWrite(TRIGG_MAX,LOW);
  //  digitalWrite(TRIGG_MIN,LOW);
  //}
}
