"""
Real-time Hand Gesture Recognition System using OpenCV and Machine Learning
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical


class HandGestureRecognizer:
    def __init__(self, model_path=None, threshold=0.7, history_frames=30):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize model parameters
        self.model = self._load_model(model_path) if model_path else None
        self.threshold = threshold
        self.gesture_labels = self._get_gesture_labels()
        self.history_frames = history_frames
        self.frame_history = []
        
        # Performance metrics
        self.fps = 0
        self.processing_time = 0
        self.last_time = time.time()
        
        # Action mapping
        self.action_mapping = {
            'thumbs_up': self._action_volume_up,
            'thumbs_down': self._action_volume_down,
            'peace': self._action_screenshot,
            'open_palm': self._action_pause,
            'fist': self._action_stop,
            'pointing': self._action_next
        }
    
    def _load_model(self, model_path):
        """Load trained model from disk"""
        try:
            return load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    def _get_gesture_labels(self):
        """Get gesture label mappings"""
        # Default gestures
        return {
            0: "thumbs_up",
            1: "thumbs_down", 
            2: "peace",
            3: "open_palm",
            4: "fist",
            5: "pointing"
        }
    
    def preprocess_frame(self, frame):
        """Preprocess the input frame for hand detection"""
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        return image
    
    def detect_hands(self, image):
        """Detect hands in the image using MediaPipe"""
        results = self.hands.process(image)
        return results
    
    def extract_hand_landmarks(self, results, image_shape):
        """Extract hand landmarks from MediaPipe results"""
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to a flat list of (x, y, z) coordinates
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    # Normalize coordinates
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand_points.extend([x, y, z])
                landmarks.append(hand_points)
        return landmarks
    
    def classify_gesture(self, landmarks):
        """Classify hand gesture using the trained model"""
        if not landmarks or not self.model:
            return None, 0.0
        
        # Update frame history with new landmarks
        self.frame_history.append(landmarks)
        if len(self.frame_history) > self.history_frames:
            self.frame_history.pop(0)
        
        # If we don't have enough frames yet, return None
        if len(self.frame_history) < self.history_frames:
            return None, 0.0
        
        # Prepare input for the model
        input_data = np.array([self.frame_history])
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        gesture_idx = np.argmax(prediction)
        confidence = prediction[gesture_idx]
        
        # Return gesture only if confidence is above threshold
        if confidence >= self.threshold:
            gesture_name = self.gesture_labels.get(gesture_idx, "unknown")
            return gesture_name, confidence
        else:
            return "unknown", confidence
    
    def draw_landmarks(self, image, results):
        """Draw hand landmarks on the image"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        return image
    
    def calculate_fps(self):
        """Calculate frames per second"""
        current_time = time.time()
        self.processing_time = current_time - self.last_time
        self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
        self.last_time = current_time
    
    def trigger_action(self, gesture_name):
        """Trigger action based on recognized gesture"""
        action_func = self.action_mapping.get(gesture_name)
        if action_func:
            action_func()
    
    def _action_volume_up(self):
        """Increase system volume"""
        print("Action: Volume Up")
        # Platform-specific code to increase volume would go here
        # For example, on Windows:
        # import pycaw.pycaw as pycaw
        # ...
    
    def _action_volume_down(self):
        """Decrease system volume"""
        print("Action: Volume Down")
        # Platform-specific code to decrease volume
    
    def _action_screenshot(self):
        """Take a screenshot"""
        print("Action: Screenshot")
        # Platform-specific screenshot code
    
    def _action_pause(self):
        """Pause/play media"""
        print("Action: Pause/Play")
        # Media control code
    
    def _action_stop(self):
        """Stop media or application"""
        print("Action: Stop")
        # Stop action code
    
    def _action_next(self):
        """Navigate to next item"""
        print("Action: Next")
        # Navigation code
    
    def process_frame(self, frame):
        """Process a single frame from the video feed"""
        # Preprocess frame
        image = self.preprocess_frame(frame)
        
        # Detect hands
        results = self.detect_hands(image)
        
        # Make image writeable again
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks = self.extract_hand_landmarks(results, image.shape)
        
        # Draw landmarks on image
        image = self.draw_landmarks(image, results)
        
        # Classify gesture if landmarks are detected
        gesture_name, confidence = None, 0
        if landmarks:
            gesture_name, confidence = self.classify_gesture(landmarks[0])  # Use first detected hand
            
            # Trigger action if gesture is recognized
            if gesture_name and gesture_name != "unknown":
                self.trigger_action(gesture_name)
        
        # Calculate FPS
        self.calculate_fps()
        
        # Display information on frame
        cv2.putText(image, f"FPS: {self.fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if gesture_name:
            cv2.putText(image, f"Gesture: {gesture_name}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image, gesture_name, confidence


class GestureModelTrainer:
    """Class for training and evaluating hand gesture recognition models"""
    
    def __init__(self, num_classes, sequence_length=30):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.model = None
    
    def build_lstm_model(self):
        """Build LSTM model for sequence classification"""
        # 21 landmarks with 3 values (x, y, z) each = 63 features
        input_shape = (self.sequence_length, 63)
        
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=16, validation_split=0.2):
        """Train the model"""
        if not self.model:
            self.build_lstm_model()
        
        # Prepare callbacks
        log_dir = os.path.join('logs', time.strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_callback = ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[tensorboard_callback, checkpoint_callback, early_stopping]
        )
        
        return history
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        if not self.model:
            print("No model to evaluate")
            return None
        
        return self.model.evaluate(X_test, y_test)


class GestureDataCollector:
    """Class for collecting training data for new gestures"""
    
    def __init__(self, output_dir='data', sequence_length=30):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.output_dir = output_dir
        self.sequence_length = sequence_length
        self.sequence_data = []
        self.recording = False
        self.current_gesture = None
        self.countdown = 0
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def start_recording(self, gesture_name, countdown=3):
        """Start recording a new gesture sequence"""
        self.current_gesture = gesture_name
        self.countdown = countdown
        self.sequence_data = []
        print(f"Get ready to record '{gesture_name}' in {countdown} seconds...")
    
    def process_frame(self, frame):
        """Process a frame for data collection"""
        # Preprocess frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        
        # Detect hands
        results = self.hands.process(image)
        
        # Make the image writeable again
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.countdown > 0:
            cv2.putText(image, str(self.countdown), (320, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 4)
            self.countdown -= 1
            return image, False
        
        if self.recording and len(self.sequence_data) < self.sequence_length:
            # Extract landmarks if hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmark data
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    self.sequence_data.append(landmarks)
                    
                    # Display progress
                    cv2.putText(image, f"Recording: {len(self.sequence_data)}/{self.sequence_length}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "No hand detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check if sequence is complete
            if len(self.sequence_data) >= self.sequence_length:
                self._save_sequence()
                return image, True
        elif not self.recording:
            cv2.putText(image, "Press 'r' to start recording", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image, False
    
    def _save_sequence(self):
        """Save recorded sequence to disk"""
        if not self.current_gesture:
            print("No gesture name specified")
            return
        
        # Create directory for gesture if it doesn't exist
        gesture_dir = os.path.join(self.output_dir, self.current_gesture)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
        
        # Generate filename based on timestamp
        filename = os.path.join(gesture_dir, f"{int(time.time())}.pkl")
        
        # Save sequence data
        with open(filename, 'wb') as f:
            pickle.dump(self.sequence_data, f)
        
        print(f"Saved sequence for '{self.current_gesture}' to {filename}")
        self.recording = False
        self.current_gesture = None
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.recording and self.current_gesture:
            self.recording = True
            self.sequence_data = []
            print(f"Started recording for '{self.current_gesture}'")
        else:
            self.recording = False
            print("Recording stopped")


class DataProcessor:
    """Class for processing collected gesture data for training"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
    
    def load_dataset(self):
        """Load and preprocess dataset from collected sequences"""
        sequences = []
        labels = []
        
        # Get list of gestures (subdirectories)
        gestures = [d for d in os.listdir(self.data_dir) 
                   if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Create label mapping
        label_map = {gesture: idx for idx, gesture in enumerate(gestures)}
        
        # Load sequences for each gesture
        for gesture_name in gestures:
            gesture_dir = os.path.join(self.data_dir, gesture_name)
            
            for filename in os.listdir(gesture_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(gesture_dir, filename)
                    
                    with open(filepath, 'rb') as f:
                        sequence = pickle.load(f)
                        sequences.append(sequence)
                        labels.append(label_map[gesture_name])
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        
        # Save label map
        with open(os.path.join(self.data_dir, 'label_map.pkl'), 'wb') as f:
            pickle.dump({v: k for k, v in label_map.items()}, f)
        
        return X, y, label_map
    
    def prepare_data(self):
        """Prepare data for training, including train-test split"""
        X, y, label_map = self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, label_map


def main():
    """Main function to run the hand gesture recognition system"""
    # Check if a trained model exists
    model_path = 'models/hand_gesture_model.h5'
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        # Initialize the hand gesture recognizer with the trained model
        recognizer = HandGestureRecognizer(model_path=model_path)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame, gesture, confidence = recognizer.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition', processed_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No trained model found. Would you like to collect training data? (y/n)")
        choice = input().lower()
        
        if choice == 'y':
            collect_training_data()
        else:
            print("Exiting program.")


def collect_training_data():
    """Function to collect training data for new gestures"""
    # Define gestures to collect
    gestures = ['thumbs_up', 'thumbs_down', 'peace', 'open_palm', 'fist', 'pointing']
    
    # Initialize data collector
    collector = GestureDataCollector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    current_gesture_index = 0
    sequence_count = 0
    max_sequences_per_gesture = 30
    
    print(f"Starting data collection for {gestures[current_gesture_index]}")
    collector.start_recording(gestures[current_gesture_index])
    
    while cap.isOpened() and current_gesture_index < len(gestures):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame for data collection
        processed_frame, sequence_complete = collector.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Data Collection', processed_frame)
        
        # Check if sequence is complete
        if sequence_complete:
            sequence_count += 1
            print(f"Completed sequence {sequence_count}/{max_sequences_per_gesture} for {gestures[current_gesture_index]}")
            
            if sequence_count >= max_sequences_per_gesture:
                current_gesture_index += 1
                sequence_count = 0
                
                if current_gesture_index < len(gestures):
                    print(f"Moving to next gesture: {gestures[current_gesture_index]}")
                    collector.start_recording(gestures[current_gesture_index])
                else:
                    print("Data collection complete!")
                    break
            else:
                # Start recording next sequence after a short delay
                time.sleep(1)
                collector.start_recording(gestures[current_gesture_index])
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            collector.toggle_recording()
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Train model with collected data
    print("Would you like to train a model with the collected data? (y/n)")
    choice = input().lower()
    if choice == 'y':
        train_model()


def train_model():
    """Function to train a model using collected data"""
    # Initialize data processor
    processor = DataProcessor()
    
    # Prepare data
    X_train, X_test, y_train, y_test, label_map = processor.prepare_data()
    
    print(f"Training with {len(X_train)} sequences, testing with {len(X_test)} sequences")
    print(f"Gesture classes: {label_map}")
    
    # Initialize and train model
    trainer = GestureModelTrainer(num_classes=len(label_map))
    trainer.build_lstm_model()
    
    # Print model summary
    trainer.model.summary()
    
    # Train model
    history = trainer.train(X_train, y_train, epochs=50)
    
    # Evaluate model
    loss, accuracy = trainer.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    trainer.save_model('models/hand_gesture_model.h5')


if __name__ == "__main__":
    main()