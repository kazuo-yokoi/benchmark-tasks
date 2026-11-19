// Trainer hand for take-a-cup-from-a-trembling-hand: the tremor and the person's letting go.
// Arduino Uno or Nano, three MG90S micro servos (5-6 V supply of their own, common ground), one microswitch.
//
//   pin 9  pan servo  (flex: the wrist's vertical axis)      +-5.5 deg
//   pin 10 tilt servo (roll: along the forearm)              +-3.8 deg, a quarter cycle behind the pan
//   pin 11 release servo (the finger block, at the knuckles) 0 = holding (light grip), -38 deg = open
//   pin 2  microswitch in the little finger's pad, closed (LOW) while the cup's base rests on it
//   pin 3  reset button (to ground): fingers close again, for the next trial
//
// The person lets go (the fingers open) once the cup's base has been off the little finger for 250 ms, the way a
// person lets go once the cup's weight is taken. The shaking never stops during a trial.

#include <Servo.h>

const float FLEX_DEG = 5.5, ROLL_DEG = 3.8;            // the simulation's FLEX_AMP and ROLL_AMP
const float HZ = 4.5;                                // task.py TREMOR_HZ
const int PAN_CENTRE = 90, TILT_CENTRE = 90, REL_HOLD = 93, REL_OPEN = 52;   // trim to the printed parts
const unsigned long LET_GO_MS = 250;

Servo pan, tilt, rel;
unsigned long t0, liftedSince = 0;
bool open_ = false;

int deg_us(float deg) { return (int)(500 + deg * 2000.0 / 180.0 + 0.5); }

void setup() {
  pinMode(2, INPUT_PULLUP);
  pinMode(3, INPUT_PULLUP);
  pan.attach(9);
  tilt.attach(10);
  rel.attach(11);
  rel.write(REL_HOLD);
  t0 = millis();
}

void loop() {
  float t = (millis() - t0) / 1000.0;
  float w = 2 * PI * HZ;
  // microseconds for sub-degree steps (about 11.1 us per degree on an MG90S)
  pan.writeMicroseconds(deg_us(PAN_CENTRE + FLEX_DEG * sin(w * t)));
  tilt.writeMicroseconds(deg_us(TILT_CENTRE + ROLL_DEG * sin(w * t + PI / 2)));

  bool resting = digitalRead(2) == LOW;
  if (!open_) {
    if (resting) liftedSince = 0;
    else if (liftedSince == 0) liftedSince = millis();
    else if (millis() - liftedSince >= LET_GO_MS) {
      open_ = true;
      rel.write(REL_OPEN);
    }
  }
  if (digitalRead(3) == LOW) {                         // reset for the next trial
    open_ = false;
    liftedSince = 0;
    rel.write(REL_HOLD);
    t0 = millis();
  }
  delay(2);                                            // about 400 updates a second
}
