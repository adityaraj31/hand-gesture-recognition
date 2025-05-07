  # Hand Gesture Recognition System

A real-time hand gesture recognition system built with Python, OpenCV, MediaPipe, and TensorFlow that can identify common hand gestures through a webcam feed.

![Hand Gesture Recognition Demo](https://via.placeholder.com/800x400?text=Hand+Gesture+Recognition+Demo)

## Features

- **Real-time hand detection and tracking** using MediaPipe
- **Recognition of 6 default gestures**:
  - Thumbs up
  - Thumbs down
  - Peace sign
  - Open palm
  - Fist
  - Pointing finger
- **Configurable action triggers** for recognized gestures
- **Low latency processing** optimized for real-time usage
- **Extensible framework** to add and train new custom gestures
- **Performance metrics** display (FPS, processing time)
- **Confidence scoring** for gesture classification

## Requirements

- Python 3.8+
- Webcam or camera device
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the System

Execute the main script to start the gesture recognition system:

```bash
python hand_gesture_recognition.py
```

- If a trained model exists, the system will launch in recognition mode
- If no model exists, you'll be prompted to collect training data

### Collecting Training Data

When prompted to collect training data:

1. Follow the on-screen instructions to record gesture sequences
2. For each gesture, perform the gesture steadily in front of the camera
3. The system will capture multiple sequences per gesture automatically
4. After data collection, you'll be prompted to train the model

### Controls During Recognition

- Press `q` to quit the application
- Actions will be triggered automatically when gestures are recognized with sufficient confidence

### Controls During Data Collection

- Press `r` to manually toggle recording mode
- Press `q` to quit the application

## System Architecture

The system consists of four main components:

1. **HandGestureRecognizer**: Main class for real-time detection and classification
2. **GestureModelTrainer**: LSTM neural network training for gesture recognition
3. **GestureDataCollector**: Tool for collecting training data for new gestures
4. **DataProcessor**: Prepares collected data for model training

### Machine Learning Approach

The gesture recognition uses a two-stage approach:

1. **Hand Detection & Landmark Extraction**: 
   - Using MediaPipe's hand tracking to identify 21 hand landmarks
   - Each landmark contains 3D coordinates (x, y, z)

2. **Gesture Classification**:
   - An LSTM neural network analyzes sequences of hand landmarks
   - Sequential information helps distinguish between similar gestures
   - Confidence scoring determines whether a gesture is recognized

## Extending the System

### Adding New Gestures

1. Modify the list of gestures in the `collect_training_data()` function
2. Run the application and select the data collection option
3. Follow the prompts to record sequences for each gesture
4. Train the model with the new data

### Customizing Actions

To customize actions triggered by gestures:

1. Open `hand_gesture_recognition.py`
2. Locate the action methods in the `HandGestureRecognizer` class (e.g., `_action_volume_up`)
3. Modify the code in these methods to perform your desired actions
4. Update the `action_mapping` dictionary to associate gestures with actions

### Model Customization

To adjust the model architecture:

1. Modify the `build_lstm_model()` method in the `GestureModelTrainer` class
2. Change hyperparameters like the number of LSTM units, dropout rate, etc.
3. Retrain the model with the updated architecture

## Performance Optimization

For better performance:

- Reduce `sequence_length` (default: 30) for lower latency but potentially reduced accuracy
- Adjust `threshold` in `HandGestureRecognizer` to balance between sensitivity and false positives
- Run on a machine with GPU support for faster inference

## Troubleshooting

### Common Issues

- **No camera detected**: Ensure your webcam is properly connected and not in use by another application
- **Low recognition accuracy**: Try retraining the model with more varied gesture samples
- **High latency**: Lower the resolution of the camera capture or optimize model parameters

### Model Training Issues

- **Training fails**: Ensure you have sufficient training data (at least 20 samples per gesture)
- **Overfitting**: Increase dropout rate or reduce model complexity
- **Underfitting**: Increase model complexity or provide more training data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking framework
- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
