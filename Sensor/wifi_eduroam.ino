#include <Adafruit_MLX90640.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <NTPClient.h>
#define EAP_IDENTITY "login"           //if connecting from another corporation, use identity@organization.domain in Eduroam
#define EAP_USERNAME "login"           //oftentimes just a repeat of the identity
#define EAP_PASSWORD "password"        //your Eduroam password


Adafruit_MLX90640 mlx;
float frame[32 * 24]; // 32x24 thermal image

const char *ssid = "eduroam";          // Eduroam SSID

const char* scriptURL = "https://script.google.com/macros/s/AKfycbzSSfJYbMKFx35IHz_aI7nBTyX5mbdvoKxHIydY9eg1M1p21xBbUfRgIzKfMvBkAf0/exec"; // Replace with your Google Apps Script URL
const String token = "ZXNwOmM2OTc5YmI5MDAyODNkNTk=";  // Token for your API

// NTP client to get current time
WiFiUDP udp;
NTPClient timeClient(udp, "pool.ntp.org", 3600, 60000); // 3 seconds offset, update every minute

int estimateTime(float dist) {
  if (dist >= 13.0) return 20;
  if (dist >= 10.4) return round(9.0 + (dist - 10.4) * (20.0 - 9.0) / (13.0 - 10.4));
  if (dist >= 7.1) return round(4.0 + (dist - 7.1) * (9.0 - 4.0) / (10.4 - 7.1));
  if (dist >= 2.7) return round((dist - 2.7) * (4.0 - 0.0) / (7.1 - 2.7));
  return 0;
}
/*
void go_to_sleep() { 
    Serial.println("ESP32 entering deep sleep for 30 seconds...");
    esp_sleep_enable_timer_wakeup(30 * 1000000); // 30 seconds in microseconds
    esp_deep_sleep_start();
}
*/

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Wire.begin(6, 7);

  Serial.println("Initializing MLX90640...");
  if (!mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("MLX90640 not found! Check wiring.");
    while (1);
  }
  mlx.setResolution(MLX90640_ADC_18BIT);
  mlx.setRefreshRate(MLX90640_4_HZ);

  WiFi.disconnect(true);  //disconnect form wifi to set new wifi connection
  WiFi.mode(WIFI_STA);    //init wifi mode
  WiFi.begin(ssid, WPA2_AUTH_PEAP, EAP_IDENTITY, EAP_USERNAME, EAP_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to Wi-Fi...");
  }
  Serial.println("Connected to Wi-Fi");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {

  const int numReadings = 20;  // Number of readings to average
  float totalDistance = 0.0;
  float max_distance = 0.0;
  
  // Take multiple readings and calculate the maximum distance for each reading
  for (int i = 0; i < numReadings; i++) {
    // Capture the thermal frame
    if (mlx.getFrame(frame) != 0) {
      Serial.println("Failed to read frame data!");
      return;
    }

    max_distance = 0.0;
    float dist = 0.0;
    
    // Process the frame to calculate the maximum distance
    for (int h = 0; h < 24; h++) {
      for (int w = 0; w < 32; w++) {
        float temperature = frame[h * 32 + w];
        dist = 0.0;
        if (w > 0 && w < 5 && h > 10 && h < 15 && temperature < 32) dist = 0.0;
        else if (h > 10 && h < 15 && temperature > 30) dist = (w + 1) * 0.4;
        if (dist > max_distance) max_distance = dist;
      }
    }

    totalDistance += max_distance;
    delay(100);  // Short delay between readings
  }

  // Compute the average distance from multiple readings
  float avgDistance = totalDistance / numReadings;

  // Estimate the time based on the average distance
  int timeEst = estimateTime(avgDistance);

  // Get current time (hour)
  timeClient.update();
  String hour = timeClient.getFormattedTime().substring(0, 5);  // Extract "HH:MM"

  // Print data
  Serial.print("Maximum Distance: ");
  Serial.print(max_distance, 1);
  Serial.print(" Waiting Time: ");
  Serial.print(timeEst, 1);
  Serial.print(" Time: ");
  Serial.print(hour);
  Serial.println();

  String jsonData = "{\"action\":\"updateWaitingTime\",";
jsonData += "\"waitingTime\":" + String(timeEst) + ",";  // 
jsonData += "\"restaurant\":\"olivier\",";
jsonData += "\"hour\":\"" + hour + "\",";  // 
jsonData += "\"token\":\"" + token + "\"}";


    // Send data via HTTP
    HTTPClient http;
    http.begin(scriptURL);
    http.addHeader("Content-Type", "application/json");
    Serial.println("Response Payload: ");
    Serial.println(jsonData);
    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0) {
        String payload = http.getString();
        Serial.print("HTTP Response Code: ");
        Serial.println(httpResponseCode);

    } else {
        Serial.print("HTTP Error Code: ");
        Serial.println(httpResponseCode);
    }

    http.end();
    delay(3000);
    /*go_to_sleep();
}
