#include <Servo.h>
#include <NewPing.h>
#include <PID_v1.h>
#define trig 6
#define echo 7

Servo brushless;
NewPing sonar(trig, echo, 400);

double speed = 0;
double distance = 3;

String serial_data;
String kp_s;
String ki_s;
String kd_s;
String sp_s;
String ts_s;

float kp = 0;
float ki = 0;
float kd = 0;
double sp = 0;
int ts = 0;

void setup() {
  Serial.begin(115200);
  brushless.attach(9);
  brushless.writeMicroseconds(1000);
  delay(1000);
}

void write_motor(int s){
  brushless.writeMicroseconds(s);
}

int getDistance(){
  delay(50);
  unsigned int uS;                      
  uS = sonar.ping();
  return uS / US_ROUNDTRIP_CM;
}

void calib(){
  brushless.writeMicroseconds(1000);
}

String getValue(String data, char separator, int index)
{
  int found = 0;
  int strIndex[] = {0, -1};
  int maxIndex = data.length()-1;

  for(int i=0; i<=maxIndex && found<=index; i++){
    if(data.charAt(i)==separator || i==maxIndex){
        found++;
        strIndex[0] = strIndex[1]+1;
        strIndex[1] = (i == maxIndex) ? i+1 : i;
    }
  }

  return found>index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void loop() {
  serial_data = "";
  if(Serial.available() > 0){
      serial_data = Serial.readString();
  }
  if(serial_data[0] == 'c'){
      //Serial.println("Calibration Mode");
      calib();
  }
  if(serial_data[0] == 'k'){
    serial_data.remove(0,2);
    kp_s = getValue(serial_data, ' ', 0);
    ki_s = getValue(serial_data, ' ', 1);
    kd_s = getValue(serial_data, ' ', 2);
    sp_s = getValue(serial_data, ' ', 3);
    ts_s = getValue(serial_data, ' ', 4);

    kp = kp_s.toFloat();
    ki = ki_s.toFloat();
    kd = kd_s.toFloat();
    sp = sp_s.toFloat();
    ts = ts_s.toInt();
    
    PID myPID(&distance, &speed, &sp, kp, ki, kd, DIRECT);
    myPID.SetTunings(kp, ki, kd);
    myPID.SetOutputLimits(1340, 1430);
    myPID.SetMode(AUTOMATIC);
    for(int i=0; i<ts; i++){
      distance = getDistance();
      
      Serial.println(distance);
      myPID.Compute();
      
      //Serial.println(speed);
      write_motor(speed);
    }
    myPID.Reset();
  }
  write_motor(1000);
}
