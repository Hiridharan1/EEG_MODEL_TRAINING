import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import scipy.io

# Load the DREAMER dataset from Kaggle path
def load_dreamer_data():
    """Load DREAMER dataset from Kaggle path"""
    data = scipy.io.loadmat('/kaggle/input/dreamer/DREAMER.mat')
    return data['DREAMER']

def prepare_eeg_data(dreamer_data):
    """
    Prepare EEG data from DREAMER dataset
    Returns DataFrame with EEG signals
    """
    big_df = pd.DataFrame()
    
    # Get EEG data for all participants
    for i in range(23):  # for each participant
        participant_data = dreamer_data['Data'][0,0][0][i][0][0][2][0][0][1]  # EEG stimuli data
        for j in range(18):  # for each video
            current_data = participant_data[j][0]
            
            # Create arrays for patient and video indices
            patient_index = [i] * len(current_data)
            video_index = [j] * len(current_data)
            
            # Get electrode readings
            electrode_data = {
                'patient_index': patient_index,
                'video_index': video_index
            }
            
            # Add electrode columns
            electrode_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                             'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            
            for k, name in enumerate(electrode_names):
                electrode_data[name] = [row[k] for row in current_data]
            
            # Create temporary DataFrame and append
            temp_df = pd.DataFrame(electrode_data)
            big_df = pd.concat([big_df, temp_df], ignore_index=True)
    
    return big_df

class EEGMusicRecommender:
    def __init__(self, eeg_data):
        self.eeg_data = eeg_data
        self.scaler = StandardScaler()
        self.features = None
        self.labels = None
        self.models = {}
        
    def extract_features(self, window_size=128):
        """Extract features from EEG signals"""
        features_list = []
        labels_list = []
        
        for (patient, video), group in self.eeg_data.groupby(['patient_index', 'video_index']):
            # Get EEG readings
            eeg_signals = group[['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                               'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']].values
            
            # Process each window
            for i in range(0, len(eeg_signals) - window_size, window_size // 2):
                window = eeg_signals[i:i + window_size]
                
                # Time domain features
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0)
                
                # Frequency domain features
                fft_vals = np.abs(np.fft.fft(window, axis=0))
                freq_features = np.mean(fft_vals, axis=0)
                
                # Combine features
                window_features = np.concatenate([mean, std, freq_features])
                features_list.append(window_features)
                labels_list.append(video)
        
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)
        
        # Scale features
        self.features = self.scaler.fit_transform(self.features)
        
        return self.features, self.labels
    
    def build_cnn_model(self):
        """Build CNN model for EEG classification"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=(self.features.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(len(np.unique(self.labels)), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train CNN, SVM, and KNN models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=random_state
        )
        
        # Prepare data for CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Train CNN
        print("Training CNN...")
        cnn_model = self.build_cnn_model()
        cnn_history = cnn_model.fit(
            X_train_cnn, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Train SVM
        print("\nTraining SVM...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=random_state)
        svm_model.fit(X_train, y_train)
        
        # Train KNN
        print("\nTraining KNN...")
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        
        # Store models
        self.models = {
            'CNN': cnn_model,
            'SVM': svm_model,
            'KNN': knn_model
        }
        
        # Get predictions
        cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
        svm_pred = svm_model.predict(X_test)
        knn_pred = knn_model.predict(X_test)
        
        # Calculate accuracies
        accuracies = {
            'CNN': accuracy_score(y_test, cnn_pred),
            'SVM': accuracy_score(y_test, svm_pred),
            'KNN': accuracy_score(y_test, knn_pred)
        }
        
        return accuracies, cnn_history
    
    def visualize_results(self, history, accuracies):
        """Visualize training results"""
        plt.figure(figsize=(15, 5))
        
        # Plot CNN training history
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('CNN Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot model comparisons
        plt.subplot(1, 2, 2)
        plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Comparisons')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the EEG music recommender"""
    print("Loading DREAMER dataset...")
    dreamer_data = load_dreamer_data()
    
    print("Preparing EEG data...")
    eeg_df = prepare_eeg_data(dreamer_data)
    print(f"Dataset shape: {eeg_df.shape}")
    
    print("\nInitializing recommender system...")
    recommender = EEGMusicRecommender(eeg_df)
    
    print("Extracting features...")
    features, labels = recommender.extract_features()
    print(f"Features shape: {features.shape}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    
    print("\nTraining models...")
    accuracies, history = recommender.train_models()
    
    print("\nModel Accuracies:")
    for model, accuracy in accuracies.items():
        print(f"{model}: {accuracy:.4f}")
    
    print("\nVisualizing results...")
    recommender.visualize_results(history, accuracies)

if __name__ == "__main__":
    main()
