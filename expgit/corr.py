import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, 1:-1]
print(df.head())
print("777777777777777777777777777777777777777777777777777777777777777")
# We don't have to do this every time.
# We select variables with high correlation to take a quick look.

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]

corr = df[num_cols].corr()
print(corr)
# It is not preferred to work with both variables that have high correlation with each other.

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(corr, cmap="RdBu")
#plt.show()

# Removing Highly Correlated Variables

cor_matrix = df.corr().abs()

triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
print(triangle_matrix)

drop_list = [col for col in triangle_matrix.columns if any(triangle_matrix[col] > 0.90)]
print(drop_list)
print("**********************************************")
print(cor_matrix[drop_list])
df.drop(drop_list, axis=1)
print(df.columns)

# Let's turn what we have done into a function

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()

    triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in triangle_matrix.columns if any(triangle_matrix[col] > 0.90)]
    
    if plot:
        sns.set(rc={"figure.figsize": (12, 12)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)
