# Fake News Detection System

A machine learning-based system to detect fake news articles using text classification techniques. This project implements multiple classification algorithms to identify misleading or false news content.

## Project Overview

This system uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify news articles as either genuine or fake. The implementation includes:

- Text preprocessing and cleaning
- Feature extraction using **TF-IDF vectorization**
- Multiple classification models for comparison
- Performance evaluation metrics
- Interactive manual testing functionality

## Dataset

The system is trained on a labeled dataset consisting of:

- `Fake.csv` - Contains fake news articles
- `True.csv` - Contains genuine news articles

Each dataset includes features like **title, text, subject, and date of publication**.

## Data Preprocessing

The text preprocessing pipeline includes:

- Converting text to lowercase
- Removing special characters, URLs, and HTML tags
- Removing punctuation and numbers
- Cleaning and normalizing text data

## Feature Extraction

Text data is transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency) vectorization**, which captures the importance of words in documents relative to the entire corpus.

## Models Implemented

- **Logistic Regression** - A statistical model for binary classification
- **Decision Tree** - A tree-based model that makes decisions based on feature values
- **Random Forest** - An ensemble of decision trees for improved performance and generalization

## Performance Evaluation

The models are evaluated using:

- Accuracy scores
- Classification reports (**Precision, Recall, F1-score**)

## Manual Testing

The system includes a function to manually test individual news articles:

- Users can input any news text to get classification results from all three models.

## Usage

1. Load the datasets (`Fake.csv` and `True.csv`)
2. Preprocess the text data
3. Split data into **training** and **testing** sets
4. Train the models
5. Evaluate performance
6. Use the **manual testing function** to classify new articles

## Requirements

To run this project, install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Results

- The models achieve high accuracy in distinguishing between fake and genuine news articles.
- The system provides a **comprehensive approach** to fake news detection by comparing predictions from multiple models.

---

