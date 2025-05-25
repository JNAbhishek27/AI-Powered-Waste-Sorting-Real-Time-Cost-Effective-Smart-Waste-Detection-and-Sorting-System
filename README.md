# AI-Powered Waste Sorting System
> Real-Time, Cost-Effective Smart Waste Detection and Sorting Solution

## 📌 Project Overview

This project is a smart AI-integrated waste management system that detects, classifies, and sorts waste in real-time using an ESP32-CAM and a lightweight machine learning model. The system is designed to promote proper waste segregation at the source, reduce human effort, and improve recycling and disposal efficiency.

## 🎯 Features

- 📷 **ESP32-CAM** captures live images of waste.
- 🤖 **Edge AI model** classifies waste into:
  - Dry waste
  - Metallic waste
  - Biomedical waste
- ♻️ **Rotating inner bin** automatically directs waste to the correct compartment.
- 🔊 **Voice output** announces the waste type.
- 🌞 **Light sensor** detects waste insertion.
- 📡 **Telegram bot** sends real-time classification and bin status updates.
- 🎁 **Reward system** (optional): awards eco-points based on waste contribution.
- 🗺️ **Smart mapping** (optional): real-time bin status monitoring on a map.

## 🛠️ Hardware Components

- ESP32-CAM Module
- Servo Motor / Stepper Motor (for rotating bin)
- Arduino UNO / Nano (for sensors and voice output)
- Light Sensor (e.g., LDR)
- Speaker or Voice Module (e.g., DFPlayer Mini)
- Power Supply (e.g., 5V adapter)
- 3D-printed or DIY rotating bin structure
- WiFi connectivity

## 🧠 Software Components

- Pre-trained image classification model (TensorFlow Lite / Edge Impulse)
- ESP32 Arduino IDE firmware (C++)
- Python/Flask server (optional for mapping/central monitoring)
- Telegram Bot (via BotFather and HTTP API)
- HTML + Leaflet.js map frontend (optional)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/JNAbhishek27/ai-waste-sorting.git
cd ai-waste-sorting
