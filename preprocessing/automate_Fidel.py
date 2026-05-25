import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocess_data():

    # load dataset
    df = pd.read_csv('stroke_raw/stroke.csv')

    # drop id
    df.drop('id', axis=1, inplace=True)

    # handling missing value
    df['bmi'].fillna(
        df['bmi'].median(),
        inplace=True
    )

    # remove duplicate
    df.drop_duplicates(inplace=True)

    # outlier handling
    numeric_cols = [
        'age',
        'avg_glucose_level',
        'bmi'
    ]

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df[col] >= lower) &
            (df[col] <= upper)
        ]

    # binning
    bins = [0, 18, 40, 60, 100]

    labels = [
        'Teen',
        'Adult',
        'Middle Age',
        'Senior'
    ]

    df['age_group'] = pd.cut(
        df['age'],
        bins=bins,
        labels=labels
    )

    # encoding
    categorical_cols = [
        'gender',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status',
        'age_group'
    ]

    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # split feature target
    X = df.drop('stroke', axis=1)

    y = df['stroke']

    # scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # processed dataframe
    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df['stroke'] = y.values

    # create output folder
    output_dir = 'preprocessing/stroke_preprocessing'

    os.makedirs(output_dir, exist_ok=True)

    # save dataset
    processed_df.to_csv(
        os.path.join(output_dir, 'stroke_clean.csv'),
        index=False
    )

    print("Preprocessing berhasil!")
    print(processed_df.head())

if __name__ == "__main__":
    preprocess_data()