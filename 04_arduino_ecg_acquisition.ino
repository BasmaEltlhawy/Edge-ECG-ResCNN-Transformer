Here's the code with all comments removed:
#include <compat/deprecated.h>
#include <FlexiTimer2.h>

#define NUMCHANNELS 6
#define HEADERLEN 4
#define PACKETLEN (NUMCHANNELS * 2 + HEADERLEN + 1)
#define SAMPFREQ 256
#define TIMER2VAL (1024/(SAMPFREQ))                    
#define LED1  13
#define CAL_SIG 9

volatile unsigned char TXBuf[PACKETLEN];
volatile unsigned char TXIndex;
volatile unsigned char CurrentCh;
volatile unsigned char counter = 0;
volatile unsigned int ADC_Value = 0;

void Toggle_LED1(void){
 if((digitalRead(LED1))==HIGH){ digitalWrite(LED1,LOW); }
 else{ digitalWrite(LED1,HIGH); }
}

void toggle_GAL_SIG(void){
 if(digitalRead(CAL_SIG) == HIGH){ digitalWrite(CAL_SIG, LOW); }
 else{ digitalWrite(CAL_SIG, HIGH); }
}

void setup() {
 noInterrupts();  
 pinMode(LED1, OUTPUT);  
 digitalWrite(LED1,LOW); 
 pinMode(CAL_SIG, OUTPUT);

 TXBuf[0] = 0xa5;    
 TXBuf[1] = 0x5a;    
 TXBuf[2] = 2;       
 TXBuf[3] = 0;       
 TXBuf[4] = 0x02;    
 TXBuf[5] = 0x00;    
 TXBuf[6] = 0x02;    
 TXBuf[7] = 0x00;    
 TXBuf[8] = 0x02;    
 TXBuf[9] = 0x00;    
 TXBuf[10] = 0x02;   
 TXBuf[11] = 0x00;   
 TXBuf[12] = 0x02;   
 TXBuf[13] = 0x00;   
 TXBuf[14] = 0x02;   
 TXBuf[15] = 0x00;  
 TXBuf[2 * NUMCHANNELS + HEADERLEN] =  0x01;  

 FlexiTimer2::set(TIMER2VAL, Timer2_Overflow_ISR);
 FlexiTimer2::start();

 Serial.begin(57600);

 interrupts();  
}

void Timer2_Overflow_ISR() {
  Toggle_LED1();
  
  for(CurrentCh=0;CurrentCh<6;CurrentCh++){
    ADC_Value = analogRead(CurrentCh);
    TXBuf[((2*CurrentCh) + HEADERLEN)] = ((unsigned char)((ADC_Value & 0xFF00) >> 8));
    TXBuf[((2*CurrentCh) + HEADERLEN + 1)] = ((unsigned char)(ADC_Value & 0x00FF)); 
  }

  for(TXIndex=0;TXIndex<17;TXIndex++){
    Serial.write(TXBuf[TXIndex]);
  }
  
  TXBuf[3]++;
  
  counter++;
  if(counter == 12){  
    counter = 0;
    toggle_GAL_SIG(); 
  }
}

void loop() {
 __asm__ __volatile__ ("sleep");
}

