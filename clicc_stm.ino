
/*
 * CLICC Timing Controller
 * Cumulative Lifetime Imaging on CMOS-Camera
 *
 * Controls LED excitation, camera gate timing, and background
 * measurements for the CLICC luminescence imaging system.
 *
 * The controller receives commands and timing parameters from
 * the Python control software via serial communication.
 *
 * Hardware:
 *   Microcontroller : STM32 Nucleo-F446RE
 *   LED trigger     : Digital pin 9
 *   Camera gate     : Digital pin 10
 *   Boost converter : PA4 (analog output)
 *
 * Serial communication:
 *   Baud rate: 9600
 *
 * Original CLICC project:
 *   https://gitlab.com/groussea/clicc
 *
 * Original authors:
 *   Gauthier Rousseau, Marianne Pons, Hessel Adelerhof,
 *   Nathan Pellerin, Mart Giesbergen, Bastien Carde,
 *   Martin Wolff, Koen Blanckaert, Sergey M. Borisov,
 *   Benoit Fond
 *
 * STM32 port and extensions:
 *   Helmar Wiltsche
 *
 * Subsequent updates and adaptations:
 *   Helmar Wiltsche and Alfred Maurer

 * Date: August 2025 (original version)
 * Last updated: March 2026
 */

// -----------------------------------------------------------------------------
// Pin definitions
// -----------------------------------------------------------------------------

// LED and camera trigger pins.
//#define PORTLED B000010    // LED/laser trigger
//#define PORTGATE B000100   // Camera gate trigger
//#define PORTOFF B000000    // All pins off

const int led_pin = 9;     // LED/laser trigger output
const int gate_pin = 10;   // Camera gate trigger output
const int boost_pin = PA4; // Analog output for boost-converter voltage control

PinName PinName_LED;
PinName PinName_Gate;

// -----------------------------------------------------------------------------
// Default timing parameters (microseconds)
// -----------------------------------------------------------------------------

int delay_cam_start = 150; // Minimum delay required by the camera between exposures
int delay_led_end;
int delay_cam_end;

// -----------------------------------------------------------------------------
// Timing parameters received via serial communication
// -----------------------------------------------------------------------------

int number_exposure;       // Number of exposures per acquisition
int t1;                    // Gate 1 delay
int t2;                    // Gate 2 delay
int t_on1;                 // LED excitation pulse duration
int vval;                  // Boost-converter control value
int deltat;                // Camera gate duration
int t2_dlr;                // Timing parameter for DLR acquisition
int t2_start;              // Optional start-time offset for Gate 2

// Serial command and loop variables
int incomingData;          // Command character received via serial
int x;                     // Exposure loop counter

/*
 * Initialization
 *
 * Configures serial communication, trigger outputs, and the
 * analog output used to control the boost converter.
 */

void setup() {
  // Initialize serial communication at 9600 baud.
  Serial.begin(9600);
  Serial.println(">>> CLICC Timing Controller ready <<<");

  // Configure trigger outputs.
  pinMode(led_pin, OUTPUT);
  pinMode(gate_pin, OUTPUT);

  PinName_LED = digitalPinToPinName(led_pin);
  PinName_Gate = digitalPinToPinName(gate_pin);

  // Configure the boost-converter control output.
  pinMode(boost_pin, OUTPUT);
  analogWriteResolution(12); // 12-bit output: 0–4095
  analogWrite(boost_pin, 0); // Initialize output to zero
}

/*
 * Main loop
 *
 * Reads incoming serial commands and their associated timing
 * parameters, then executes the requested acquisition sequence.
 */

void loop() {
  // Check for incoming serial data.
  if (Serial.available() > 0) {
    // Read the command character.
    incomingData = Serial.read();

    // Allow additional serial data to arrive before parsing.
    // At 9600 baud, transmitting one byte takes approximately 1 ms.
    delayMicroseconds(2000);

    // Read and parse the timing parameters if data is available.
    if (Serial.available() > 0) {
      String all_num = Serial.readStringUntil(' ');
      sscanf(all_num.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d", &number_exposure, &deltat, &t2_dlr, &t_on1, &delay_led_end, &delay_cam_end, &vval, &t2_start);
    }

    // Execute the acquisition sequence selected by the command.
    switch(incomingData) {
      case '1':
        acquireGate1();
        break;
      case '2':
        acquireGate2();
        break;
      case '3':
        backgroundGate1();
        break;
      case '4':
        backgroundGate2();
        break;
      case '5':
        acquireGate1_DLR();
        break;
      case '6':
        acquireGate2_DLR();
        break;
    }
  }
}

/*
 * Gate 1 acquisition
 *
 * For each exposure:
 *   1. Activate the LED for t_on1.
 *   2. Wait for the configured LED-to-camera delay.
 *   3. Activate the camera gate for deltat.
 *   4. Wait before starting the next exposure.
 */

void acquireGate1() {

  analogWrite(boost_pin, vval);

  for(x = 0; x < number_exposure; x++) {
    // Activate the LED.
    digitalWriteFast(PinName_LED, HIGH);
    delayMicroseconds(t_on1);

    // Deactivate the LED and wait before opening the camera gate.
    digitalWriteFast(PinName_LED, LOW);
    delayMicroseconds(delay_led_end);

    // Open the camera gate.
    digitalWriteFast(PinName_Gate, HIGH);
    delayMicroseconds(deltat);

    // Close the camera gate and wait for the next exposure.
    digitalWriteFast(PinName_Gate, LOW);
    delayMicroseconds(delay_cam_end + deltat);
  }
}

/*
 * Gate 2 acquisition
 *
 * Uses the second acquisition timing sequence. The optional t2_start
 * parameter adjusts the delay before and after the camera gate.
 */

void acquireGate2() {

  analogWrite(boost_pin, vval);

  for(x = 0; x < number_exposure; x++) {
    // Activate the LED.
    digitalWriteFast(PinName_LED, HIGH);
    delayMicroseconds(t_on1);

    // Deactivate the LED and wait for the configured gate delay.
    digitalWriteFast(PinName_LED, LOW);
    if (t2_start == 0) {
      delayMicroseconds(delay_led_end + deltat);
    }
    else {
      delayMicroseconds(delay_led_end + t2_start);
    }

    // Open the camera gate.
    digitalWriteFast(PinName_Gate, HIGH);
    delayMicroseconds(deltat);

    // Close the camera gate and wait for the next exposure.
    digitalWriteFast(PinName_Gate, LOW);

    if (t2_start == 0) {
      delayMicroseconds(delay_cam_end);
    }
    else {
      delayMicroseconds(delay_cam_end + deltat - t2_start);
    }
  }

}

/*
 * Gate 1 background acquisition
 *
 * Acquires background images without LED excitation. Only the
 * camera gate is activated during each exposure.
 */

void backgroundGate1() {
  for(x = 0; x < number_exposure; x++) {
    // Ensure the LED is off.
    digitalWriteFast(PinName_LED, LOW);

    // Wait before opening the camera gate.
    delayMicroseconds(delay_cam_start);

    // Open and close the camera gate.
    digitalWriteFast(PinName_Gate, HIGH);
    delayMicroseconds(deltat);

    digitalWriteFast(PinName_Gate, LOW);
  }

}

/*
 * Gate 2 background acquisition
 *
 * Acquires background images without LED excitation using the
 * Gate 2 background timing sequence.
 */

void backgroundGate2() {
  for(x = 0; x < number_exposure; x++) {
    // Wait before opening the camera gate.
    delayMicroseconds(delay_cam_start);

    // Open and close the camera gate.
    digitalWriteFast(PinName_Gate, HIGH);
    delayMicroseconds(deltat);

    digitalWriteFast(PinName_Gate, LOW);
  }

}

/*
 * Gate 1 acquisition using Dual Lifetime Referencing (DLR)
 *
 * The camera gate opens while the LED is active. The gate duration
 * and remaining LED pulse duration are determined by deltat and t_on1.
 */

void acquireGate1_DLR() {

  analogWrite(boost_pin, vval);

  for(x = 0; x < number_exposure; x++) {
    // Activate the LED and open the camera gate.
    digitalWriteFast(PinName_LED, HIGH);
    digitalWriteFast(PinName_Gate, HIGH);

    delayMicroseconds(deltat);

    // Close the camera gate and complete the LED pulse.
    digitalWriteFast(PinName_Gate, LOW);
    delayMicroseconds(t_on1 - deltat);

    // Deactivate the LED and wait before the next exposure.
    digitalWriteFast(PinName_LED, LOW);

    delayMicroseconds(delay_cam_end + t2_dlr + delay_led_end);
  }

}

/*
 * Gate 2 acquisition using Dual Lifetime Referencing (DLR)
 *
 * Activates the LED, waits for the configured delay, and then
 * opens the camera gate for t2_dlr microseconds.
 */

void acquireGate2_DLR() {

  analogWrite(boost_pin, vval);

  for(x = 0; x < number_exposure; x++) {
    // Activate the LED.
    digitalWriteFast(PinName_LED, HIGH);
    delayMicroseconds(t_on1);

    // Deactivate the LED and wait before opening the camera gate.
    digitalWriteFast(PinName_LED, LOW);
    delayMicroseconds(delay_led_end);

    // Open the camera gate for the configured DLR interval.
    digitalWriteFast(PinName_Gate, HIGH);
    delayMicroseconds(t2_dlr);

    // Close the camera gate and wait for the next exposure.
    digitalWriteFast(PinName_Gate, LOW);
    delayMicroseconds(delay_cam_end);
  }

}
