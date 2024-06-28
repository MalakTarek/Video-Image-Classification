# Video-Image-Classification
Project 1: Image Classification with SVM and CNN

This project involves building image classification models using SVM and CNN to classify images of cats and dogs.

    Libraries and Modules:
        Installed libraries: OpenCV, Numpy, Pandas, Matplotlib, Scikit-learn, TensorFlow.
        Imported necessary modules.

    Data Collection and Preprocessing:
        Defined constants: image size (128x128), number of clusters (100).
        Loaded and preprocessed images from the specified folder.
        Split data into training, validation, and test sets.
        Performed data augmentation using ImageDataGenerator.

    Feature Extraction with SIFT and Bag-of-Words:
        Extracted SIFT features.
        Built a vocabulary using KMeans clustering.
        Computed Bag-of-Words histograms.

    Standardization:
        Standardized BOW features using StandardScaler.

    SVM Classifier:
        Trained an SVM classifier with GridSearchCV to find the best parameters.
        Evaluated the SVM model on the test set.

    CNN Model:
        Created and trained a CNN model with augmented data.
        Evaluated the CNN model on the test set.

    Comparison and Analysis:
        Compared SVM and CNN models based on accuracy, precision, recall, and F1-score.
        Analyzed the performance of both models.
        Trained and evaluated CNN models with different optimizers (Adam, SGD, RMSprop).

Project 2: Video Classification with InceptionV3 and LSTM

This project involves classifying videos from the UCF101 dataset using InceptionV3 for feature extraction and LSTM for sequence modeling.

    Libraries and Modules:
        Installed libraries: Imageio, OpenCV, TensorFlow, TF-Hub.
        Imported necessary modules.

    Data Collection and Preprocessing:
        Listed and fetched videos from the UCF101 dataset.
        Created a DataFrame with video names and their corresponding class labels.
        Split the dataset into training, validation, and test sets.

    Feature Extraction with InceptionV3:
        Initialized InceptionV3 for feature extraction.
        Extracted features from video frames using InceptionV3.

    Label Encoding and One-Hot Encoding:
        Encoded labels using LabelEncoder.
        Converted integer labels to one-hot encoded vectors.

    RNN Model (LSTM):
        Defined and compiled an LSTM model.
        Trained the LSTM model on extracted features.
        Evaluated the LSTM model on the test set.
