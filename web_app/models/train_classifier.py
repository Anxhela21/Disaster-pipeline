#TRAIN DATA:

import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# def load_data(database_filepath):
#     engine = create_engine('sqlite:///'+database_filepath)
#     df = pd.read_sql_table(database_filepath, engine)
#     X = df['message']
#     y = df.iloc[:, 4:]
#     category_names = list(df.columns[4:]) 
#     return X, y, category_names


def load_data(database_filepath):
    
    engine = create_engine("sqlite:///%s"%database_filepath)
    df = pd.read_sql_table('Disaster', engine)
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    print(y.head(1))
    y=y.astype(int)
    category_names = y.columns
    return X, y, category_names


    
    
     #This is the tokenizing stage in the pipeline 
def tokenize(text):
    
    stop_words = stopwords.words("english")
        
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    # lemmatize and remove stop words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]


    return words



def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

#     parameters = {'clf__estimator__n_estimators': [50, 100],
#                   'clf__estimator__min_samples_split': [2, 3, 4],
#                   'clf__estimator__criterion': ['entropy', 'gini']
#                  }
#     cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline




def evaluate_model(model, X_test, y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
    
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))


    
def save_model(model, model_filepath):
    filename = 'finalized_model.p'
    pickle.dump(model, open(model_filepath, 'wb'))
    
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()