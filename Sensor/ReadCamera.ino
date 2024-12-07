#include <Adafruit_MLX90640.h>
#include <Wire.h>
Adafruit_MLX90640 mlx;
float frame[32 * 24]; // 32x24 thermal image

void setup() {
  Serial.begin(115200); // Initialize serial communication
  while (!Serial) delay(10);
  Wire.begin(6, 7);

  Serial.println("Initializing MLX90640...");

  // Initialize the sensor
  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("MLX90640 not found! Check wiring.");
    while (1);
  }
  Serial.println("MLX90640 initialized successfully.");

  // Set parameters for resolution and refresh rate
  mlx.setResolution(MLX90640_ADC_18BIT); // High resolution
  mlx.setRefreshRate(MLX90640_4_HZ);    // 2 Hz refresh rate
}

void loop() {
  
  // Capture the thermal frame
  if (mlx.getFrame(frame) != 0) {
    Serial.println("Failed to read frame data!");
    return;
  }

  Serial.println("Temperature Data (32x24):");
  for (int h = 0; h < 24; h++) {
    for (int w = 0; w < 32; w++) {
      float temperature = frame[h * 32 + w];
      Serial.print(temperature, 1); // Print temperature with 1 decimal place
      Serial.print("\t"); // Tab for separation
    }
    Serial.println(); // Newline after each row
  }

delay(1000); // Wait before reading the next frame
  
}
