import pandas as pd #package to work with CSV tables
import time
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('training_set_outcome_identification.csv', sep='|')
Xtrain = df['extracted_text'].tolist()
Xtrain = [i.replace('\n', '')[-2500:] for i in Xtrain]
Ytrain = df['Uitspraak'].tolist()

pipeline = Pipeline([
    ('charvec', CountVectorizer(analyzer='char')),
    ('clf', LinearSVC())
])

parameters = {
    'charvec__ngram_range': [(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,2),(2,3),(2,4),(2,5),(2,6),(3,3),(3,4),(3,5),(3,6),(4,4),(4,5),(4,6),(5,5),(5,6),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(1,8),(2,8),(3,8),(4,8),(5,8),(6,8),(7,8),(8,8)],
    'charvec__lowercase': (True, False),
    'charvec__max_df': (0.7, 0.8, 0.9, 1.0), # ignore words that occur as more than 1% of corpus
    'charvec__binary': (False, True),
    'charvec__max_features': (None, 2000, 3000, 5000),
    'clf__C':(0.001, 0.01, 0.1, 1, 10, 100)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1, cv=3)
t0 = time.time()
grid_search.fit(Xtrain, Ytrain)

print("done in %0.3fs" % (time.time() - t0))
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
    
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))