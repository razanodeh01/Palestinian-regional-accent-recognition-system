"""
Group members:
1- Razan Abdelrahman - 1200531
2- Duaa Abu Sliman - 1200909
3- Safaa taweel - 1202065

"""

import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import warnings
from scipy.io.wavfile import WavFileWarning

# Ignore WavFileWarnings
warnings.simplefilter('ignore', WavFileWarning)

def read_wav_file(filename):
    try:
        sampling_freq, audio = wav.read(filename)
        if audio.ndim > 1:  # Convert stereo to mono
            audio = np.mean(audio, axis=1)
        if audio.size == 0:
            raise ValueError("Audio file is empty")
        return sampling_freq, audio
    except Exception as e:
        raise IOError(f"Could not read file {filename}: {e}")

def extract_features(directory):
    features = []
    labels = []
    num_files = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.wav'):
                filepath = os.path.join(root, filename)
                try:
                    sampling_freq, audio = read_wav_file(filepath)
                    mfcc_features = mfcc(audio, sampling_freq, numcep=20, nfft=2048)
                    if mfcc_features.size > 0:
                        features.append(np.mean(mfcc_features, axis=0))
                        labels.append(root.split(os.sep)[-1])
                    num_files += 1
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
    return features, labels, num_files

def predict(model, scaler, features):
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)

def train_svm(features, labels, C=1, kernel='rbf'):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    svm_model = SVC(C=C, kernel=kernel, random_state=42)
    svm_model.fit(features_scaled, labels)
    return svm_model, scaler

def train_rf(features, labels, n_estimators=100):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(features_scaled, labels)
    return rf_model, scaler

def train_knn(features, labels, num_neighbors=8):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn_model.fit(features_scaled, labels)
    return knn_model, scaler

def file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hides the small tkinter window
    return filedialog.askopenfilename()

def print_classification_report_table(report_dict):
    # Convert classification report dict to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    print("\n", report_df.to_string(), "\n")

def main_menu(train_features, train_labels, test_features, test_labels, prediction_file_path):
    models_info = []
    
    while True:
        print("\n**************************************Acoustic Palestinian regional accent recognition system*************************************\n")
        print("Welcome to our accent recognition system, Choose the model you want to train the data with:")
        print("1: Support Vector Machines 'SVM'")
        print("2: Random Forest 'RF'")
        print("3: K-Nearest Neighbors Algorithm 'KNN'")
        print("4: Print Classification Reports for All Models")
        print("5: Print Confusion Matrix for All Models")
        print("6: Exit")
        choice = input("Please enter your choice (1, 2, 3, 4, 5, or 6): ")

        if choice == '6':
            print("\nThank you for using our system!\n")
            break

        if choice == '1':
            model, scaler = train_svm(train_features, train_labels)
            model_name = "SVM"
        elif choice == '2':
            model, scaler = train_rf(train_features, train_labels)
            model_name = "RF"
        elif choice == '3':
            model, scaler = train_knn(train_features, train_labels)
            model_name = "KNN"
        elif choice == '4':
            for model_info in models_info:
                print(f"\nClassification Report for {model_info['name']}:\n")
                print_classification_report_table(model_info['classification_report'])
            continue
        elif choice == '5':
            for model_info in models_info:
                print(f"\nConfusion Matrix for {model_info['name']}:\n")
                print(model_info['confusion_matrix'])
            continue
        else:
            print("\nInvalid choice, please try again.\n")
            continue

        # Evaluate the model
        test_predictions = predict(model, scaler, test_features)
        accuracy = accuracy_score(test_labels, test_predictions)
        report = classification_report(test_labels, test_predictions, output_dict=True)
        conf_matrix = confusion_matrix(test_labels, test_predictions)
        
        models_info.append({
            'name': model_name, 
            'classification_report': report,
            'confusion_matrix': conf_matrix
        })

        print(f"\nAccuracy of {model_name}: {accuracy * 100:.2f}%")

        # Use the prediction file path to predict accent
        try:
            sampling_freq, audio = read_wav_file(prediction_file_path)
            mfcc_features = mfcc(audio, sampling_freq, numcep=20, nfft=2048)
            mfcc_features = np.mean(mfcc_features, axis=0).reshape(1, -1)
            predicted_label = predict(model, scaler, mfcc_features)
            print(f"\nPredicted accent: {predicted_label[0]}\n")
        except Exception as e:
            print(f"\nError processing the prediction: {str(e)}\n")

if __name__ == "__main__":
    train_data_path = "C:/Users/Support/Desktop/4th_Second/SPOKEN/PROJECT/project_solution/Dataset/Training data"
    test_data_path = "C:/Users/Support/Desktop/4th_Second/SPOKEN/PROJECT/project_solution/Dataset/Testing data"
    prediction_file_path = file_dialog()

    # Load and process training and testing data at the start
    print("\nLoading and processing training data...")
    train_features, train_labels, _ = extract_features(train_data_path)
    print("Done, The training data is loaded.\n")
    print("Loading and processing testing data...")
    test_features, test_labels, _ = extract_features(test_data_path)
    print("Done, The testing data is loaded.\n")

    main_menu(train_features, train_labels, test_features, test_labels, prediction_file_path)
