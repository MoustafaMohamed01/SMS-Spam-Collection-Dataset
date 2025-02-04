# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
nltk.download("punkt")
from nltk.corpus import stopwords
stopwords.words("english")
import string
string.punctuation
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import timeit
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
# -
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import pickle
# --------

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " " .join(y)

# -----------

def transform_text_2(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove non-alphanumeric tokens (punctuation & special characters)
    filtered_tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords
    final_tokens = [word for word in filtered_tokens if word not in stopwords.words("english")]
    
    # Apply stemming
    stemmed_tokens = [ps.stem(word) for word in final_tokens]
    
    # Return as a space-separated string
    return " ".join(stemmed_tokens)

# --------------------------

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    return accuracy,precision
    

# ---------

# Import Dataset
df = pd.read_csv("spam.csv",encoding="latin1")
print(df.head(),"\n")

print("Dataframe shape: ",df.shape,"\n")

# Project Main Steps
# 1. Data cleaning
# 2. EDA
# 3. Tezt Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deployment 

# 1.Data Cleaning
print(df.info(),"\n")

# Check for null values
print("null values: ",df.isnull().sum(),"\n")

# Drop cols ["Unnamed: 2" , "Unnamed: 3" , "Unnamed: 4"]
df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
print(df.head(1),"\n")

# Rename Columns
df.rename(columns={"v1":"target","v2":"text"},inplace=True)
print(df.head(1),"\n")

encoder = LabelEncoder()
encoder.fit_transform(df["target"])

df["target"] = encoder.fit_transform(df["target"])

print("(target) column data type: ",df["target"].dtypes,"\n")
print(df.head(1))

# Check for duplicate values
print("duplicate values: ",df.duplicated().sum(),"\n")

# Remove duplicates
df = df.drop_duplicates(keep="first")
print("duplicate values: ",df.duplicated().sum(),"\n")

print(df.info(),"\n")

# 2.EDA
print(df["target"].value_counts())

# target coulumn values count pie graph
explode = (0.1, 0)
plt.pie(df["target"].value_counts(),labels=["ham","spam"],autopct="%0.2f",colors=sns.color_palette("pastel"),explode=explode,shadow=True)

# create a number of charahcters column
df["characters_num"] = df["text"].apply(len)
print(df.head(1))

# create a number of words column
df["text"].apply(lambda x: nltk.word_tokenize(x))
df["text"].apply(lambda x: len(nltk.word_tokenize(x)))
df["words_num"] = df["text"].apply(lambda x: len(nltk.word_tokenize(x)))
print(df.head(1))

# create a number of sentences column
df["text"].apply(lambda x: nltk.sent_tokenize(x))
df["text"].apply(lambda x: len(nltk.sent_tokenize(x)))
df["sentences_num"] = df["text"].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df.head(1))

# descriptive statistics of columns["characters_num","words_num","sentences_num"]
print(df[["characters_num","words_num","sentences_num"]].describe())

# ham descriptive statistics of columns["characters_num","words_num","sentences_num"]
print(df[df["target"]==0][["characters_num","words_num","sentences_num"]].describe())

# spam descriptive statistics of columns["characters_num","words_num","sentences_num"]
print(df[df["target"]==1][["characters_num","words_num","sentences_num"]].describe())

# Graph showing (ham.spam) SMS messages by number of characters
plt.figure(figsize=(10,5),dpi=150)
sns.histplot(df[df["target"]==0]["characters_num"],color="blue")
sns.histplot(df[df["target"]==1]["characters_num"],color="red",alpha=0.5)
plt.legend(title="Target", labels=["Ham", "Spam"])
plt.grid(True, linestyle="--")
plt.show()

# Graph showing (ham.spam) SMS messages by number of words
plt.figure(figsize=(10,5),dpi=150)
sns.histplot(df[df["target"]==0]["words_num"],color="blue")
sns.histplot(df[df["target"]==1]["words_num"],color="red",alpha=0.5)
plt.legend(title="Target", labels=["Ham", "Spam"])
plt.grid(True, linestyle="--")
plt.show()

# Graph showing (ham.spam) SMS messages by number of sentences
plt.figure(figsize=(10,5),dpi=150)
sns.histplot(df[df["target"]==0]["sentences_num"],color="blue")
sns.histplot(df[df["target"]==1]["sentences_num"],color="red",alpha=0.5)
plt.legend(title="Target", labels=["Ham", "Spam"])
plt.grid(True, linestyle="--")
plt.show()

sns.pairplot(df,hue="target",palette="Set2")
plt.show()

# correlation
print(df.corr(numeric_only=True))

# heatmap shows correlation
sns.heatmap(df.corr(numeric_only=True),cmap='Blues',annot=True)
plt.show()


# 3.Data Preprocessing
# Lower case
# Tokenization
# Removing special characters
# Removing stop characters
# Stemming

ps = PorterStemmer()

# --------------------------------------------
import timeit

time1 = timeit.timeit(lambda: df["text"].apply(transform_text),number=1)

time2 = timeit.timeit(lambda: df["text"].apply(transform_text_2),number=1)

print(f"Runtime for transform_text: {time1:.5f} seconds")
print(f"Runtime for transform_text_2: {time2:.5f} seconds")
# --------------------------------------------

df["transformed_text"] = df["text"].apply(transform_text_2)
print(df.head(1))


wc = WordCloud(width=500,height=500,min_font_size=10,background_color="white")

spam_wc = wc.generate(df[df["target"]==1]["transformed_text"].str.cat(sep=" "))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df["target"]==0]["transformed_text"].str.cat(sep=" "))
plt.imshow(ham_wc)

spam_corpus = []
for msg in df[df["target"]==1]["transformed_text"].tolist():
    for words in msg.split():
        spam_corpus.append(words)

print(len(spam_corpus))

print(Counter(spam_corpus).most_common(30))

# convert it to Dataframe
print(pd.DataFrame(Counter(spam_corpus).most_common(30)))

# graph for most popular spam corpus
plt.figure(figsize=(10,6),dpi=150)
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1],palette="Set2")
plt.xlabel("Words")
plt.ylabel("Popularity")
plt.xticks(rotation=90)
plt.show()


ham_corpus = []
for msg in df[df["target"]==0]["transformed_text"].tolist():
    for words in msg.split():
        ham_corpus.append(words)

print(len(ham_corpus))
print(pd.DataFrame(Counter(ham_corpus).most_common(30)))

# graph for most popular ham corpus
plt.figure(figsize=(10,6),dpi=150)
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1],palette="Set2")
plt.xlabel("Words")
plt.ylabel("Popularity")
plt.xticks(rotation=90)
plt.show()


# 4.Model Building
# Text Vectorization
# using bag of words

cv = CountVectorizer()

X = cv.fit_transform(df["transformed_text"]).toarray()
print(X.shape)

y = df["target"].values
print(y.shape)

# Train , Split the Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_score(y_test,y_pred_gnb)
confusion_matrix(y_test,y_pred_gnb)
precision_score(y_test,y_pred_gnb)

# MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred_mnb = mnb.predict(X_test)
accuracy_score(y_test,y_pred_mnb)
confusion_matrix(y_test,y_pred_mnb)
precision_score(y_test,y_pred_mnb)

# BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
y_pred_bnb = mnb.predict(X_test)
accuracy_score(y_test,y_pred_bnb)
confusion_matrix(y_test,y_pred_bnb)
precision_score(y_test,y_pred_bnb)


# Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer (TfidfVectorizer)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["transformed_text"]).toarray()
print(X.shape)

y = df["target"].values
print(y.shape)

# Train , Split the Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_score(y_test,y_pred_gnb)
confusion_matrix(y_test,y_pred_gnb)
precision_score(y_test,y_pred_gnb)

# MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred_mnb = mnb.predict(X_test)
accuracy_score(y_test,y_pred_mnb)
confusion_matrix(y_test,y_pred_mnb)
precision_score(y_test,y_pred_mnb)

# BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
y_pred_bnb = mnb.predict(X_test)
accuracy_score(y_test,y_pred_bnb)
confusion_matrix(y_test,y_pred_bnb)
precision_score(y_test,y_pred_bnb)

svc = SVC(kernel="sigmoid",gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver="liblinear",penalty="l1")
rfc = RandomForestClassifier(n_estimators=50,random_state=2)
abc = AdaBoostClassifier(n_estimators=50,random_state=2,algorithm="SAMME")
bc = BaggingClassifier(n_estimators=50,random_state=2)
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    "SVC":svc,
    "KN" :knc,
    "NB" :mnb,
    "DT" :dtc,
    "LR" :lrc,
    "RF" :rfc,
    "AdaBoost":abc,
    "BgC":bc,
    "ETC":etc,
    "GBDT":gbdt,
    "xgb":xgb
}

# train classfier
train_classifier(svc,X_train,y_train,X_test,y_test)

accuracy_scores = []
precision_scores = []
for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    print("For: ",name)
    print("accuracy: ",current_accuracy)
    print("precision: ",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# create a dataframe for algorithms performace
performance_df = pd.DataFrame({"Algorithm":clfs.keys(),"Accuracy":accuracy_scores,"Precision":precision_scores})
print(performance_df)

# melting the dataframe
performance_df1 = pd.melt(performance_df,id_vars="Algorithm")
print(performance_df1)

# graph shows each algorith (accuracy,precision)
sns.barplot(data=performance_df1,x="Algorithm",y="value",hue="variable")
plt.legend(loc=(1.1,0.5))
plt.ylim(0.5,1.0)
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()

# 5.Model Improve
# Change max_features parameter of TfIdf

tfidf = TfidfVectorizer(max_features=3000)

temp_df = pd.DataFrame({"Algorithm": clfs.keys(), "Accuracy_max_ft_3000": accuracy_scores, "Precision_max_ft_3000": precision_scores})
print(temp_df)

# merge Dataframes
new_df_max_ft = performance_df.merge(temp_df,on="Algorithm")
print(new_df_max_ft)

X = tfidf.fit_transform(df["transformed_text"]).toarray()

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
y = df["target"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred_gnb))
print(confusion_matrix(y_test,y_pred_gnb))
print(precision_score(y_test,y_pred_gnb))

# MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred_mnb = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred_mnb))
print(confusion_matrix(y_test,y_pred_mnb))
print(precision_score(y_test,y_pred_mnb))

# BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
y_pred_bnb = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred_bnb))
print(confusion_matrix(y_test,y_pred_bnb))
print(precision_score(y_test,y_pred_bnb))


accuracy_scores = []
precision_scores = []
for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    print("For: ",name)
    print("accuracy: ",current_accuracy)
    print("precision: ",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

temp_df_scaled = pd.DataFrame({"Algorithm":clfs.keys(),"Accuracy_scaled":accuracy_scores,"Precision_scaled":precision_scores})
print(temp_df_scaled)

# Merge Dataframes
new_df_scaled = new_df_max_ft.merge(temp_df_scaled,on="Algorithm")
print(new_df_scaled)


X = np.hstack((X,df["characters_num"].values.reshape(-1,1)))
print(X.shape)

y = df["target"].values
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred_gnb))
print(confusion_matrix(y_test,y_pred_gnb))
print(precision_score(y_test,y_pred_gnb))

# MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred_mnb = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred_mnb))
print(confusion_matrix(y_test,y_pred_mnb))
print(precision_score(y_test,y_pred_mnb))

# BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
y_pred_bnb = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred_bnb))
print(confusion_matrix(y_test,y_pred_bnb))
print(precision_score(y_test,y_pred_bnb))


accuracy_scores = []
precision_scores = []
for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    print("For: ",name)
    print("accuracy: ",current_accuracy)
    print("precision: ",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

df_num_chars = pd.DataFrame({"Algorithm":clfs.keys(),"Accuracy_num_chars":accuracy_scores,"Precision_num_chars":precision_scores})
print(df_num_chars)

# Merge Dataframes
new_df_scaled_chars = new_df_scaled.merge(df_num_chars,on="Algorithm")
print(new_df_scaled_chars)

# Voting Classifier
svc = SVC(kernel="sigmoid",gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)

voting = VotingClassifier(estimators=[("svm",svc),("nb",mnb),("et",etc)],voting="soft")

voting.fit(X_train,y_train)

y_pred = voting.predict(X_test)
print("accuracy: ",accuracy_score(y_test,y_pred))
print("precision: ",precision_score(y_test,y_pred))

# Applaying stacking
estimators = [("svm",svc),("nb",mnb),("et",etc)]
final_estimator = RandomForestClassifier()

clf = StackingClassifier(estimators=estimators,final_estimator=final_estimator)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("accuracy: ",accuracy_score(y_test,y_pred))
print("precision: ",precision_score(y_test,y_pred))

pickle.dump(tfidf,open("vectorizer.pkl","wb"))
pickle.dump(mnb,open("model.pkl","wb"))
