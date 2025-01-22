import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Taking a general look
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
print(df.head())

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

print(df[["age", "fare"]].describe().T)

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]
print(num_cols)

num_cols = [col for col in df.columns if col not in cat_cols]
print("///////////////////////////////////////")
print(num_cols)
"""
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")   

for col in num_cols:
    print("+++++++++++++++++++++++++")
    num_summary(df, col)
"""
# Let's add a few more features to this function

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[numerical_col].describe(quantiles).T)
    
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        
num_summary(df, "age", plot=True)
