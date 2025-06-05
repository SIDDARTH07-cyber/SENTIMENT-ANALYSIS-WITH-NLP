# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: P VISHNU SIDDARTH

INTERN ID: CT04DF2078

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

##This project demonstrates how to build a sentiment analysis model to classify movie reviews as positive or negative using machine learning techniques in Python. The dataset used is the popular IMDb movie review dataset, which contains 50,000 movie reviews labeled with sentiments.

Overview
Sentiment analysis, also known as opinion mining, is a powerful natural language processing (NLP) technique used to determine the emotional tone behind words. It is widely applied to understand customer opinions, reviews, social media feedback, and more.

In this project, we leverage logistic regression—a simple yet effective classification algorithm—to predict the sentiment of movie reviews. The reviews are preprocessed and converted into numerical features using TF-IDF vectorization, which captures the importance of words relative to the dataset.

Dataset
The dataset used for this project is the IMDb Dataset, which contains 50,000 movie reviews evenly split between positive and negative sentiments. Each review is a text document of varying length, often containing HTML tags and punctuation.

The dataset is read from a CSV file named 'IMDB Dataset.csv'. The main columns of interest are:

review: The movie review text.

sentiment: The label indicating if the review is positive or negative.

Data Preprocessing
Before feeding the text data into the machine learning model, the data undergoes several preprocessing steps to clean and normalize the text for better learning:

Lowercasing: All text is converted to lowercase to avoid duplication of words with different cases (e.g., "Good" and "good").

Removing HTML Tags: Many reviews contain HTML tags, which are removed using regular expressions to keep only the actual review text.

Removing Non-Alphabetical Characters: Numbers, punctuation, and special characters are removed so that only alphabets remain. This helps reduce noise and focus on meaningful words.

Removing Extra Whitespaces: Multiple spaces are replaced with a single space, and leading/trailing spaces are stripped for clean text.

Encoding Sentiments: The sentiment labels are converted to binary numeric form: 'positive' is mapped to 1 and 'negative' to 0, enabling the classification model to understand the labels.

Feature Extraction with TF-IDF
Machine learning models cannot directly work with text; hence, the text needs to be transformed into numerical features.

We use TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer to convert the cleaned text into feature vectors. TF-IDF not only counts word occurrences but also scales them by how unique they are across the dataset, reducing the weight of common words and highlighting significant terms.

stop_words='english': Removes common English words like "the", "is", "and" which do not contribute much meaning.

max_df=0.7: Ignores words that appear in more than 70% of documents, considering them too common to be useful.

This vectorization creates a sparse matrix representation of the text data for both training and testing sets.

Model Training and Evaluation
We split the dataset into training and testing subsets with an 80-20 split, ensuring that the model learns from most data and is evaluated on unseen data.

A Logistic Regression model is trained on the TF-IDF features of the training reviews to learn the patterns of positive and negative sentiments.

The trained model is then used to predict sentiments on the test data.

Metrics Used:
Accuracy: The proportion of correctly predicted sentiments.

Classification Report: Provides precision, recall, and F1-score for both classes.

Confusion Matrix: Visual representation of true vs predicted labels, highlighting true positives, true negatives, false positives, and false negatives.

A heatmap of the confusion matrix is plotted for an intuitive visualization.

Results
The model achieves high accuracy on the test set, demonstrating the effectiveness of logistic regression combined with TF-IDF features for binary sentiment classification on movie reviews.

The classification report provides insight into the balance between precision and recall, showing the model's robustness in detecting both positive and negative sentiments.

The confusion matrix heatmap visually confirms the model's strong predictive capability, with most reviews correctly classified.

Conclusion
This project successfully builds a sentiment analysis pipeline using Python, pandas, scikit-learn, and visualization tools. The key takeaways include:

Importance of thorough text preprocessing to clean noisy data.

Effectiveness of TF-IDF vectorization to extract meaningful features from text.

Logistic regression as a simple yet powerful baseline model for binary text classification.

Evaluation metrics provide a comprehensive understanding of model performance.

This project can be extended by exploring more advanced models such as Support Vector Machines, Random Forests, or deep learning approaches like LSTM and Transformers for improved accuracy.

#OUTPUT

![Image](https://github.com/user-attachments/assets/fc7226d8-f914-4621-98c8-96646b767e2d)

![Image](https://github.com/user-attachments/assets/dcbc0933-36a9-48dc-b1e1-2f69745953b9)



