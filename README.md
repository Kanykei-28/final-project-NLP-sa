# EPAM Final Project
--- 
## 1. Data Science Part
### 1.1 Problem Statement
In this project, we try to implement NLP with supervised model to classify movie review into negative and positive classes. 
</br>

### 1.2 Methodology
The DS part of this was performed in `notebooks/01_EDA.ipynb`. It follows the following pipeline: 
1. **EDA:** exploring the train.csv dataset 
2. **Model training and choosing the best model:** Training three different models on the dataset with only essential preprocessing steps (proprecced only using TF-IDF including tokenization and lowercasing, nothing else is applied), so that I could focus only on the effect of the model and choose the best model among those three as the baseline for preprocessing steps later. 
3. **Preprocessing options and choosing the best method:** After choosing the best model, we test different preprocessing methods. This way we can see any change in performance that comes from preprocessing not from the model, and select the best preprocessing pipeline. We test the following components:
    - Tokenization and basic cleaning: converting all text to lowercase, removing HTML tags and ther unnecessary symbols, working with negations (changing “can’t” to “can not”)
    - Stop-word removal: removing unnesecary words like articles, prepositions, etc, but keepig the negations like "no" and "not"
    - Comparing Stemming vs. lemmatization: comparing these tw and choosing the best one for our project 
    - Comparing vectorization methods: testing and comparing TF-IDF, CountVectorizer and HashingVectorizer. 
4. **Model tuning:** applying gridSerachCV with 3fold cross validation to tune hyperarameters of the best pipeline and model. 
5. **Final train:** selecting the best pipeline and the model, training it and saving the model. 

### 1.3 EDA: 
From EDA, we see several points:
1. Class is balanced in the dataset. So, accuracy can go as the main performance metric. Since it is not a clinical domain problem, there is no need to focus on one class more that on another, so we can just look overall a F1-score instead of looking at preision or recall separatively. Thus, accuracy will be our main metric, while F1 score will be secondary. 
2. There are no missing values in the train dataset, so no need for imputation logic.
3. From the review length discribution analysis, we can see that these are full movie reviews. Also, there is not really a big difference in terms of the length between positive and negative reviews, meaning that Length is not a strong discriminative feature. 
4. We also have a high variance in document length. Sparse vectorization methods like TF-IDF can handle this, so this method serves as a good baseline. 

### 1.3 Model Selection:
As mentioned above, we first trained three different models with the minimum preprocessing steps in orer to see which model would perform the best and select that model for later steps like testing which preprocessing pipeline provides the best results. 
All models were trained and evaluated on the same train–validation split and preprocessing pipeline (tokenization + TF-IDF vectorization), and accuracy was the main metric for comparison. The models and their performance are following:
| Model                    | Accuracy |
|--------------------------|----------|
| LinearSVC                | 0.9164   |
| LogisticReg              | 0.9008   |
| Multinomial Naive Bayes  | 0.8895   |

As LinearSVC had the higest accuracy, it was chosen for the later steps.


### 1.4 Preprocessing Pipeline Selection: 
As required in the project description, different preprocessing techniques have been tested and compared. We tried the following things: 
1. *Tokenization* done inside the pipeline of TF-IDF (it is basically the basline preprocessing pipeline)
2. *Stop-words filtering.* At this point, we will remove stopwords with the help of standard stopwords list. However, we keep negotionations like "no", "not", 'never', etc. Also, we will transform shortened words like can't and don't to can not and do not in order to highglight the difference between positive and negative tones. We will then apply LinearSVC to and compare it with the baseline pipelin to see if we need stop-words filtering or not. 
3. In order to compare *stemming VS lemmalization* and choose the best one between them, we will use two different pipelines with each of the techniques separately and compare their performance. 
4. As required in the description, we will *test at least 2 different vectorization techqniues* and compare their performnce. 

| Model       | Preprocessing Method  | Accuracy   |
|-------------|-----------------------|------------|
| LinearSVC   | Baseline (TF-IDF)     | 0.9164     |
| LinearSVC   | Stop words filtering  | 0.909125   |
| LinearSVC   | Stemming              | 0.9140     |
| LinearSVC   | Lemmalization         | 0.9149     |
| LinearSVC   | Count Vectorization   | 0.898625   |
| LinearSVC   | Hashed Vectorization  | 0.0.904875 |

From the above table, we see that the baseline pipeline (using TF-IDF vetorization only without anything else) leads to the higest accuracy. So, it was decided to take TF-IDF+LinearSVC as the final pipeline. 

### 1.4 Hyperparameter Tuning:
We used 3fold cross validation and applied GridSearchCV only on the training split and keep a separate hold-out validation set for the final score, so that there is no bias and data leakage. The results were following: 

Best CV accuracy: 0.9108749578324224
Best params: {'clf__C': 1.0, 'clf__class_weight': None, 'clf__max_iter': 5000, 'tfidf__max_df': 0.9, 'tfidf__max_features': 200000, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 2), 'tfidf__sublinear_tf': True}
Hold-out set accuracy: 0.916375

Hyperparameter tuning with GridSearchCV showed that the initial baseline pipeline with default hyperparameters was already optimal (acuracy was merely the same). Moreover, as the best CV accuracy and hold-out set accuracy were almost the same, it indicated there was no overfitting. For later, we kept those tuned hyperparameters found by GridearchCV.
Overall, the final piepline was following: 
- Tokenization
- TF-IDF vectorization
- LinearSVC 
This pipeline achieved approximately 91.6% accuracy, which is above the required 0.85 threshold.

### 1.5 Potential business applications and value for business;
This project can be used to automatically analyze customer reviews, feedback forms, or social media comments. Instead of  reading tones of texts by tehmselves, companies can quickly see whether opinions are positive or negative. But of course this classification model will not be enough to gain a deep feedback as it does not provide any context or reason why someone decided that move is good or bad. But at least it can be used to gain statistical insights about bad and good reviews at first, aka "fast review sorter" for the starter point.

From a business perspective, this helps to save time, track customer satisfaction over time, and find better decisions for product improvements. Thus, by automating text analysis companies save time and react faster to negative feedback which can improve customer experience and protect the its reputation.


---
## 2. MLE Part

### 2.1 Project Structure
The project follows following structure:

- src/train/ – training pipeline and Dockerfile for training and saving the model
- src/inference/ – inference piepline and Dockerfile for batch prediction
- src/utils/ – files contain paths, constants, preprocessing
- tests/ – smoke tests
- requirements.txt – dependencies needed for training/inference 
- requirements-dev.txt – additional libraries for notebooks and development (these were separated in order to not interfere docker reproducability and make sure no unexpected errors occur when working with Docker)
- Note: data and outputs are not included by Git as required in the project description

### 2.2 Data
The dataset is expected inside `data/raw`. The data/ directory is ignored by Git as asked in the description. So, it must be mounted into Docker when running containers.

### 2.3 Train 
Model is trained and saved in `src/train/train.py`. It validates the input data by checking that all required columns are present, that there are no missing values and labels are correct. The model is built using TF-IDF vectorization with LinearSVC. After training, the trained model is saved to `outputs/models/` and the training metrics are saved to `outputs/metrics/train_metrics.json`. All file paths are defined in src/utils/paths.py, so the code does not rely on any absolute system paths.
```markdown
Train locally by:

```bash
python -m src.train.train
```
Artifacts will be saved to `outputs/models/` and `outputs/metrics/train_metrics.json`. 

- Train using docker by: 
``` docker build -f src/train/Dockerfile -t epam_sentiment_train .
```
Run container: 
``` docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  epam_sentiment_train
  ```

### 2.4 Inference 
The inference check is implemented in `src/inference/run_inference.py`. It requires only the review text column, while the target (sentiment) column is optional. If ground-truth labels are provided, the script computes evaluation metrics. If not provided, then it saves only the predictions. For each inference run predictions and metrics are saved with timestamps. All performance logs with times can be found in and predictions.csv.

- The run_inference.py file can be run locally by using command below: 
``` python -m src.inference.run_inference --input data/raw/inference.csv
```
Outputs are saved to `outputs/predictions/` and  `outputs/metrics/`. 

- Run inference using Docker:
``` docker build -f src/inference/Dockerfile -t epam_sentiment_infer .
```
Run container: 
``` docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  epam_sentiment_infer \
  --input data/raw/inference.csv
```
Note: model must already exist in outputs/models/ to run it using Docker. 
In general, here it is imprtant to note that data directory is mounted as read-only, while the outputs directory is mounted with write permissions. No data is stored inside the container itself. Also, all generated artifacts remain on the local environemnt.

### 2.5 Testing
Simple smoke tests are implemented in `tests/test_smoke.py`. 
Ti run tests use:
``` pytest -q
```




