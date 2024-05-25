import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv('fake_or_real_news.csv')

df = df[['text', 'label']]

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=1000))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='fake')
recall = recall_score(y_test, y_pred, pos_label='fake')
f1 = f1_score(y_test, y_pred, pos_label='fake')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

log_reg = model.named_steps['logisticregression']
tfidf = model.named_steps['tfidfvectorizer']

feature_names = tfidf.get_feature_names_out()
coefficients = log_reg.coef_[0]

top_features = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
top_features = top_features.sort_values(by='coefficient', ascending=False).head(10)
print("Top features for predicting fake news:")
print(top_features)
