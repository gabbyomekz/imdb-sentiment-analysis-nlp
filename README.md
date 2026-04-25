# 🎬 IMDB Sentiment Analysis (NLP Project)

This project focuses on sentiment analysis of movie reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews as **positive** or **negative**, helping to understand audience perception and automate opinion mining at scale.

---

## 📌 Project Overview

With the explosion of user-generated content, analyzing sentiment from text data has become essential for businesses and platforms. In this project, I built and evaluated multiple machine learning and deep learning models to classify IMDB movie reviews into binary sentiment categories.

This project demonstrates:
- Text preprocessing and feature engineering
- Traditional machine learning approaches (naive bayes, k-Means, svm, logistic regression, random forest)
- Deep learning models (LSTM, GRU)
- Model evaluation and comparison

---

## 📂 Dataset

- **Dataset:** IMDB Movie Reviews Dataset  
- **Source:** The dataset was obtained from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).  
- **Size:** 50,000 reviews  
- **Classes:**  
  - Positive  
  - Negative  

### Features:
- `review`: Textual movie review  
- `sentiment`: Label (positive/negative)  

---

## 🎯 Objectives

- Clean and preprocess raw text data
- Convert text into numerical representations
- Train and evaluate multiple models
- Compare traditional ML vs deep learning approaches

---

## ⚙️ Tools & Technologies

- **Programming Language:** Python  
- **Libraries:**
  - `pandas`, `numpy` – data handling  
  - `scikit-learn` – ML models & evaluation  
  - `nltk` / `re` – text preprocessing  
  - `tensorflow / keras` – deep learning models
  - `matplotlib`, `seaborn` – visualization

# NLTK data downloads needed

```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
---

## 🧩 Project Workflow

### 1️⃣ Data Preprocessing
- Lowercasing text
- Removing punctuation and special characters
- Tokenization
- Stopword removal
- Padding sequences (for deep learning models)

---

### 2️⃣ Feature Engineering
- **TF-IDF Vectorization** for traditional models
- **Tokenization + Padding** for neural networks

---

### 3️⃣ Models Implemented

#### 🔹 Traditional Machine Learning
- Naive Bayes, svm, logistic regression, random forest Classifiers   
- K-Means Clustering (unsupervised exploration)

#### 🔹 Deep Learning Models
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

#### 🔹 Hyperparameter Tuning 
- Optimized batch size, epoch, embedding dimension and number of hidden units for GRU/LSTM were tweaked to see if performance improves
- Dropout regulation of 0.2-0.5 was used in RNN layers to prevent overfitting

---

## 📊 Model Evaluation

Models were evaluated using:
- Accuracy
- F1-Score
- Loss

---

## 📈 Results Summary

### 📊 Traditional Machine Learning Model Performance

| Model                     | Learning Type | Accuracy (%) | F1 Score (%) |
|--------------------------|--------------|--------------|--------------|
| **Logistic Regression**  | Supervised   | **88.27**    | **88.39**    |
| Support Vector Machines  | Supervised   | 87.69        | 87.77        |
| Naïve Bayes              | Supervised   | 85.62        | 85.80        |
| Random Forest            | Supervised   | 83.59        | 83.49        |
| k-Means                  | Unsupervised | 63.74        | 68.02        |

### 🔍 Key Observations

- Logistic Regression achieved the highest overall performance among traditional models.
- Support Vector Machines performed comparably, showing strong classification capability.
- Naïve Bayes provided solid baseline performance with lower computational cost.
- Random Forest underperformed slightly, possibly due to text feature sparsity.
- k-Means, being unsupervised, showed significantly lower performance, highlighting the advantage of labeled data in sentiment classification.

---

### 🤖 Deep Machine Learning Model Performance (GRU/LSTM)

| Model | Optimizer | Units | Batch Size | Dropout Rate | Embedding Size | Epochs | Accuracy | Loss   |
|---------|----------|-------|------------|--------------|----------------|--------|------------|------------|
| **GRU** | Adam     | 64    | 64         | 0.3          | 64             | 5      | **0.8870** | **0.2783** |
| LSTM    | Adam     | 64    | 64         | 0.3          | 64             | 5      | 0.7047     | 0.6201     |

### 🔍 Key Observations

- The GRU model significantly outperformed the LSTM model in both accuracy and loss.
- GRU achieved an accuracy of 88.70%, making it competitive with traditional machine learning models.
- The lower loss value for GRU (0.2783) indicates better model convergence and generalization.
- LSTM showed weaker performance under the same hyperparameter settings, suggesting it may require further tuning.
- GRU's simpler architecture likely contributed to faster learning and improved efficiency on this dataset.

---

### 🤖 Hyperparameter Tuning (GRU/LSTM)


| Model | Optimizer | Units | Batch Size | Dropout Rate | Embedding Size | Epochs | Accuracy | Loss   |
|----------|----------|----------|-------------|--------------|-----------------|---------|------------|------------|
| **GRU**  | Adam     | 64,64    | 64          | 0.5          | 64              | 10      | **0.8879** | **0.2864** |
| GRU      | Adam     | 64,64    | 128         | 0.5          | 128             | 10      | 0.5107     | 0.6927     |
| LSTM     | Adam     | 64,64    | 64          | 0.5          | 64              | 10      | 0.5112     | 0.6928     |
| **LSTM** | Adam     | 64,64    | 128         | 0.5          | 128             | 10      | **0.8773** | **0.3048** |

### 🔍 Key Observations

- Hyperparameter tuning applied on these deep machine learning models didn’t superbly increase performance. However, slight improvements are observed for:
- GRU (batch size=64, epochs=10, units=[64,64], embedding size=64) with 88.79% accuracy
- LSTM (batch size=128, epochs=10, units=[64,64], embedding size=128) with 87.33% accuracy.

---

## 📊 Overall Model Performance (Traditional and Tuned Deep Learning Models)

| Model                     | Model Type              | Accuracy (%) | Metric Value |
|--------------------------|-------------------------|--------------|--------------|
| **GRU**                  | Deep Learning           | **88.79**    | 0.2864 (Loss) |
| Logistic Regression      | Traditional (Supervised)| 88.27        | 88.39 (F1)   |
| LSTM                     | Deep Learning           | 87.73        | 0.3048 (Loss) |
| Support Vector Machines  | Traditional (Supervised)| 87.69        | 87.77 (F1)   |
| Naïve Bayes              | Traditional (Supervised)| 85.62        | 85.80 (F1)   |
| Random Forest            | Traditional (Supervised)| 83.59        | 83.49 (F1)   |
| k-Means                  | Traditional (Unsupervised)| 63.74      | 68.02 (F1)   |

> ⚠️ Note: Traditional machine learning models are evaluated using F1 Score, while deep learning models are evaluated using Loss due to differences in training and evaluation approaches.

---

## 💡 Key Insights

- The GRU model achieved the highest overall accuracy (88.79%), outperforming both traditional and other deep learning models on complex language patterns.
- Logistic Regression performed competitively, making it a strong baseline model despite its simplicity.
- LSTM showed slightly lower performance than GRU, suggesting that GRU may be more efficient for this dataset.
- Traditional models such as SVM and Naïve Bayes delivered stable and reliable results with lower computational cost.
- k-Means, being an unsupervised model, performed significantly worse, highlighting the importance of labeled data in sentiment classification tasks.
- Overall, deep learning models provided marginal accuracy improvements but at a higher computational cost compared to traditional approaches.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/gabbyomekz/imdb-sentiment-analysis-nlp.git
cd imdb-sentiment-analysis-nlp
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### 3️⃣ Run the Notebook
```bash
jupyter notebook imdb_sentiment_analysis.ipynb
```

---

## 🔮 Future Improvements

- Use of transformer model such as BERT
- Use pre-trained word embeddings (GloVe, Word2Vec)
- Deploy model as a web application (Streamlit/Flask)
- Add explainability (LIME/SHAP)

---

## 🧠 Business Applications

- Movie recommendation systems
- Customer feedback analysis
- Social media sentiment tracking
- Brand reputation monitoring

---

## 👨‍💻 Author

Developed by [Gabriel Omeke]

📧 Contact: [gabrielomeke92@gmail.com](mailto:gabrielomeke92@gmail.com)

🔗 GitHub: [github.com/gabbyomekz](https://github.com/gabbyomekz)

---








