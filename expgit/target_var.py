import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")


# Target Variable Analysis
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # Docstring
    """_summary_
    Returns the categorical, numerical, and cardinal variables in the data set

    Args:
        dataframe (_type_): _description_:
              We take variable names here
        cat_th (int, optional): _description_. Defaults to 10.
              Class threshold value for numeric but categorical variables
        car_th (int, optional): _description_. Defaults to 20.
              Class threshold value for categorical but cardinal variables
    """
    
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float"]]


    num_cols = [col for col in df.columns if col not in cat_cols]
    
    print(f"observations: {dataframe.shape[0]}")
    print(f"variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")


grab_col_names(df)


# Let's investigate why people survived. What influenced them?


# Let's see how the survived variable relates to other variables
print("****************************")
print(df.groupby("sex")["survived"].mean()) # For example, we could see this with groupby.


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"target_mean": dataframe.groupby(categorical_col)[target].mean()}))


"""
target_summary_with_cat(df, "survived", "pclass")
"""
for col in cat_cols:
    target_summary_with_cat(df, "survived", col)



# Analysis of Target Variable with Numerical Variables


print(df.groupby("survived")["age"].mean())

print(df.groupby("survived").agg({"age": "mean"}))
