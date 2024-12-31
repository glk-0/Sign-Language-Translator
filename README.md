# Sign-Language-Translator

A real-time Sign Language Translator that uses **Machine Learning** and **OpenCV** to recognize American Sign Language (ASL) gestures from a webcam feed. The project processes hand gestures, predicts the corresponding ASL letter, and displays it on the screen.

## Features ✨
- Real-time hand gesture recognition using OpenCV and MediaPipe.
- Neural network trained on 30,000+ ASL gesture images.
- Achieved **92% accuracy** on training data and **100% accuracy** on the test dataset.
- Supports real-time webcam input with minimal latency.

---

## Installation 🚀

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/sign-language-translator.git
cd sign-language-translator
```
### 2. Install Requirements
Use `pip` to install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Usage 🖥️
### 1. Run the Application
Launch the real-time ASL translator:
```bash
python read.py
```
### 2. Interact with the Webcam
- Ensure proper lighting and position your hand within the camera's view.
- The predicted ASL letter will be displayed on the screen.
- 
## How It Works 🔍
### Hand Detection:
Utilizes MediaPipe Hands to detect hand landmarks in real-time.

### Preprocessing:
Crops the hand region and resizes it to match the model's input size (64x64 pixels).

### Prediction:
Feeds the preprocessed image to the trained TensorFlow model to predict the corresponding ASL letter.

### Display Output:
Displays the predicted letter on the webcam feed.
