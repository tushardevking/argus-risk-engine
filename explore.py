import pandas as pd

def explore_data(filepath, target_column):
    """Kisi bhi dataset ko load karke basic exploration deta hai.
    filepath: CSV ka path
    target_column: label column ka naam (jaise 'Class')
    """
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nTarget distribution:")
    print(df[target_column].value_counts(normalize=True))
    return df

def data_healthcheck(df):
    print(df.info())
    print(df.describe())
    return df

# Use
df = explore_data("creditcard.csv", "Class")
data_healthcheck(df)