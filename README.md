# Deploying SPAM Classifier to Web using Streamlit
This program deploys a trained SPAM classification Machine Learning model onto a web interface using Streamlit.
The program accepts user input, preprocesses it, vectorizes it using a pre-trained TfidfVectorizer, and makes predictions using a pre-trained Extra Trees model for classifying spam messages.


### How to run

`pip install streamlit`
`streamlit run spam_classifier.py`

### Data Loading & General Summary
- texts  Spam Detection Dataset sourced from Kaggle. 
- csv file containing 5172 randomly picked texts s and their respective labels for spam or not-spam classification

### Data Pre-Processing
- renaming columns
- dropping columns with high number of null values
- Encoding target variable
- Dropping duplicate rows

### Text Processing 
- Engineering new features:
  - 'length': length of texts  message
  - 'num_words': # of words in the message
  - 'num_sent': # of sentences
- Text Cleaning using NLTK package
  - Tokenization
    - breaks down the text into individual words or tokens. 
    - divides the text into meaningful units, allowing for further analysis at the word level
  - Lowercase conversion
    - ensures that words with different capitalizations are treated as the same, reducing redundancy and ensuring consistency.
    - improves word matching and retrieval
  - Removal of non-alphanumeric characters
    - filters out symbols, punctuation marks, and special characters from the text which do not add to overall meaning of text
    - reduces noise
  - Remove of stopwords
    - commonly used words that do not add to meaning or context
    - reduces dimensionality and noise of data
  - Stemming
    - reduces words to their base or root form
    - collapses different variations of a word to a common representation
    - helps reduce the vocabulary size and improve information retrieval

### Plotting & Exploratory Data Analysis
- distributions of 'length', 'num_words', and 'num_sent' against spam and non-spam messages
  - visualize difference between predictors and outcome
  - better understanding of key differences between spam and non-spam messages:
    - On avergage SPAM texts are 67 characters longer, contain ~10 more words, and have 1 more sentance than non-SPAM text. 
  - wordclouds
    - visual representations of spam and non-spam text data where the size of each word corresponds to its frequency or importance in the text
    - SPAM wordcloud:
      - 'free'
      - 'call'
      - 'text'
      - 'offer'
      - ''win
    - Non-SPAM wordcloud:
      - 'go'
      - 'love'
      - 'come'
      - 'time'
    
### Model Inputs
- Converting cleaned text to vector inputs using Bag of Words Technique
  - 'Tfidvectorizer' converts text data into numerical feature vectors
  - calculates relative importance of each word in a document or corpus by considering its frequency in the document and its rarity in the entire corpus.
  - transformers data into a sparse matrix represenation
- Splitting data into train/test 
  - Training set shape: (4135, 6708) (4135,)
  - Test set shape: (1034, 6708) (1034,)

### Model Building
- Defining classifiers:
  - Log Regression
  - SVM
  - Multinomial Naive Bayers
  - Decision Tree
  - K-Nearest Neighbors
  - Random Forest
  - Extra Trees
  - AdaBoost
  - Bagging Classifier
  - Gradient Boosting
  - XGBoost
- Training models & reporting classification metrics
- Selecting best model
  - Extra Trees Model highest F-1 score of 91.5%
  - XGBoost 2nd highest, Random Forest 3rd highest F-1 scores
- Tuning Extra Trees model using grid search
  - New F-1 score 93.3%, up by ~2%

### Intrepretting Classification Metrics
- Accuracy:
  - Accuracy measures the overall correctness of the classifier's predictions. 
  - It represents the proportion of correctly classified instances (both spam and non-spam) out of the total number of instances. However, accuracy alone may not be sufficient when dealing with imbalanced datasets, where the number of non-spam instances significantly outweighs the number of spam instances.
- Precision:
  - Precision calculates the proportion of correctly classified spam instances out of the total instances predicted as spam. 
  - It indicates the classifier's ability to correctly identify spam messages. A high precision value means that the classifier has a low false positive rate, i.e., it is less likely to classify non-spam messages as spam. 
- Recall (Sensitivity or True Positive Rate):
  - Recall measures the proportion of correctly classified spam instances out of the total actual spam instances.
  - It represents the classifier's ability to capture all spam messages. A high recall value indicates that the classifier has a low false negative rate, i.e., it is less likely to miss actual spam messages.
- F1 Score:
  - Harmonic mean of precision and recall
  - It provides a single metric that balances both precision and recall
  - A high F1 score indicates a good balance between precision and recall
- Confusion Matrix:
  - Provides a detailed breakdown of the classifier's performance by showing the counts of true positive, true negative, false positive, and false negative predictions.

### Results
- There is a tradeoff between Recall and Precision, and this tradeoff needs to be considered.
  - Increasing the classification threshold: 
    - If you raise the classification threshold, the classifier becomes more conservative in labeling a text as spam. 
    - This can lead to a higher precision, as it reduces the chances of false positives (non-spam messages being classified as spam). 
    - However, it may result in lower recall because some actual spam messages might be incorrectly classified as non-spam and end up in the inbox.
  - Reducing the classification threshold:
    - If you lower the classification threshold, the classifier becomes more sensitive in identifying spam. 
    - This can increase recall, as more spam messages are correctly classified. 
    - However, it may lead to lower precision because the classifier is more likely to classify non-spam messages as spam, resulting in more false positives.
  - For this project, Precision takes precedence over Recall due to the significant chaos and damage a SPAM message can cause. 

### Room For Improvement
- Collect more data from external sources
  - Collect demographic, geographic data
  - domain names
  - timestamp of messages
- Try advanced unsupervised algorithms like Neural Nets to improve performance
- Seasonality analysis
- Create web interface using Streamlit where users can paste their messages to identify if they are spam or not
