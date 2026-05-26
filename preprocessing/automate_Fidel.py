import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

def preprocess_data():

    print("Memulai preprocessing...")

    # ==================================================
    # LOAD DATASET
    # ==================================================

    df = pd.read_csv(
        'mental_raw/mental.csv'
    )

    print("Dataset berhasil dimuat")
    print("Shape awal:", df.shape)

    # ==================================================
    # MISSING VALUE HANDLING
    # ==================================================

    numeric_cols = [
        'age',
        'stress_level',
        'sleep_hours',
        'physical_activity_days',
        'depression_score',
        'anxiety_score',
        'social_support_score',
        'productivity_score'
    ]

    for col in numeric_cols:

        df[col].fillna(
            df[col].median(),
            inplace=True
        )

    print("Missing value berhasil ditangani")

    # ==================================================
    # REMOVE DUPLICATE
    # ==================================================

    before = df.shape[0]

    df.drop_duplicates(
        inplace=True
    )

    after = df.shape[0]

    print(
        f"Duplicate dihapus: {before - after}"
    )

    # ==================================================
    # OUTLIER HANDLING
    # ==================================================

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

    print("Outlier berhasil ditangani")
    print("Shape setelah outlier:", df.shape)

    # ==================================================
    # BINNING
    # ==================================================

    bins = [0, 18, 30, 45, 60, 100]

    labels = [
        'Teen',
        'Young Adult',
        'Adult',
        'Middle Age',
        'Senior'
    ]

    df['age_group'] = pd.cut(
        df['age'],
        bins=bins,
        labels=labels
    )

    print("Binning age berhasil")

    # ==================================================
    # ENCODING
    # ==================================================

    categorical_cols = [
        'gender',
        'employment_status',
        'work_environment',
        'mental_health_history',
        'seeks_treatment',
        'mental_health_risk',
        'age_group'
    ]

    le = LabelEncoder()

    for col in categorical_cols:

        df[col] = le.fit_transform(
            df[col]
        )

    print("Encoding berhasil")

    # ==================================================
    # SPLIT FEATURE TARGET
    # ==================================================

    X = df.drop(
        'mental_health_risk',
        axis=1
    )

    y = df[
        'mental_health_risk'
    ]

    # ==================================================
    # SCALING
    # ==================================================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ==================================================
    # DATAFRAME FINAL
    # ==================================================

    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df[
        'mental_health_risk'
    ] = y.values

    # ==================================================
    # SAVE DATASET
    # ==================================================

    processed_df.to_csv(
        'preprocessing/mental_preprocessing/mental_clean.csv',
        index=False
    )

    print("Dataset preprocessing berhasil disimpan")

    print(
        "Lokasi: preprocessing/mental_preprocessing/mental_clean.csv"
    )

    print("Preprocessing selesai!")

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    preprocess_data()