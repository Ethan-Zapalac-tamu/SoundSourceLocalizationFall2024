#include <TMRpcm.h>
#include <pcmConfig.h>
#include <pcmRF.h>

int analog_pin_mic0 = A0; // Change these depending on the physical setup
int digital_pin_mic0 = 3;
int sound_limit = 1000; // length of sound file

void setup() {
  // 9600 is the preffered plotter serial value
  Serial.begin(9600);
  pinMode(digital_pin_mic0, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  int analog_signal = analogRead(analog_pin_mic0);
  int digital_signal = digitalRead(digital_pin_mic0);

  // The serial plotter is set up to graph the values printed to the serial monitor
  Serial.println(analog_signal);

  // Known issue, it is difficult to make a distinct threshold level for a specific voice compared to background noise
  // Light the built in LED when the digital signal is high to not affect analog plot
  if (digital_signal == HIGH) {
    digitalWrite(LED_BUILTIN, HIGH);
  } 
  else {
    digitalWrite(LED_BUILTIN, LOW);
  }

  // Delay so the serial interface isn't overwhealmed
  delay(20);
}
