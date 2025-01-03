
# Hate Speech and Offensive Language Detection

## Overview
This project is part of the **Natural Language Processing (NLP)** coursework and focuses on detecting hate speech and offensive language in social media posts. 
It combines advanced NLP preprocessing techniques with various machine learning models to classify text into categories like hate speech, offensive language, or neutral.

## Features
1. Preprocessing with tokenization, stemming, and stopword removal using NLTK.
2. Feature extraction using TF-IDF vectorization.
3. Machine learning models: Logistic Regression, Random Forest, and Support Vector Machines (SVM).
4. Data visualization for insights into class distribution and confusion matrices.
5. Performance comparison of models using precision, recall, F1-score, and accuracy metrics.

## Prerequisites
- Python 3.7 or later
- Required libraries (see `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Hate-Speech-Detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Hate-Speech-Detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your dataset in the project directory. Ensure it is named `path_to_file.csv` or update the code with your dataset's path.
2. Run the notebook `hate-speech-detection-nlp-enhanced.ipynb` in Jupyter or any compatible environment.
3. Review the outputs, including visualizations, evaluation metrics, and model comparisons.

## Dataset
This project uses a Kaggle dataset with labeled text data. The dataset includes categories for hate speech, offensive language, and neutral content.

## Results
The project compares multiple machine learning models, providing insights into their strengths and identifying the best-performing approach.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements
- Python libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn, NLTK.
- Kaggle for providing the dataset.
