# Fake News Detection System
A machine learning-based system to detect fake news articles using text classification techniques. This project implements multiple classification algorithms to identify misleading or false news content.
Project Overview
This system uses Natural Language Processing and Machine Learning techniques to classify news articles as either genuine or fake. The implementation includes:
Text preprocessing and cleaning
Feature extraction using TF-IDF vectorization
Multiple classification models for comparison
Performance evaluation metrics
Interactive manual testing functionality
Dataset
The system is trained on a labeled dataset consisting of:
Fake.csv: Contains fake news articles
True.csv: Contains genuine news articles
Each dataset includes features like title, text, subject, and date of publication.
Data Preprocessing
The text preprocessing pipeline includes:
Converting text to lowercase
Removing special characters, URLs, and HTML tags
Removing punctuation and numbers
Cleaning and normalizing text data
Feature Extraction
Text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which captures the importance of words in documents relative to the entire corpus.
Models Implemented
Logistic Regression: A statistical model for binary classification
Decision Tree: A tree-based model that makes decisions based on feature values
Random Forest: An ensemble of decision trees for improved performance and generalization
Performance Evaluation
The models are evaluated using:
Accuracy scores
Classification reports (precision, recall, F1-score)
Manual Testing
The system includes a function to manually test individual news articles:
Usage
Load the datasets (Fake.csv and True.csv)
Preprocess the text data
Split data into training and testing sets
Train the models
Evaluate performance
Use the manual testing function to classify new articles
Example:
Requirements
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
Regular expressions (re)
Results
The models achieve high accuracy in distinguishing between fake and genuine news articles. The system provides a comprehensive approach to fake news detection by comparing predictions from multiple models.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
Dataset sources
References to papers or techniques used
Any other acknowledgments or credits
