# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import nltk
import nltk
nltk.download('stopwords')
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
# -


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
