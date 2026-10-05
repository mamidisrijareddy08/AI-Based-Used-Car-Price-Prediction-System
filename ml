# ============================================================
# AI-BASED USED CAR PRICE PREDICTION SYSTEM
# Using NLP + Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
import warnings
import joblib


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

print("=" * 70)
print("AI-BASED USED CAR PRICE PREDICTION SYSTEM")
print("=" * 70)


# ============================================================
# 1. FIND AND LOAD DATASET
# ============================================================

csv_files = glob.glob("*.csv")

if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV file found. Please upload the 900-record CSV to Colab."
    )

print("\nCSV files found:")

for file in csv_files:
    print("-", os.path.basename(file))

# Find the 900-record dataset
dataset_file = None

for file in csv_files:
    if "seller_description" in os.path.basename(file).lower():
        dataset_file = file
        break

if dataset_file is None:
    dataset_file = csv_files[0]

print("\nUsing dataset:")
print(os.path.basename(dataset_file))

df = pd.read_csv(dataset_file)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET COLUMNS")
print("=" * 70)

print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "car_name",
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "seller_description",
    "selling_price"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("\nAll required columns are present!")


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nRecords before removing duplicates:", before)
print("Records after removing duplicates:", after)
print("Duplicates removed:", before - after)


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

for column in df.columns:

    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna("Unknown")

print("\nMissing values handled successfully!")

print("\nRemaining missing values:")
print(df.isnull().sum())


# ============================================================
# 9. CHECK SELLER DESCRIPTION
# ============================================================

print("\n" + "=" * 70)
print("SELLER DESCRIPTION")
print("=" * 70)

print(
    df[
        [
            "car_name",
            "seller_description",
            "selling_price"
        ]
    ].head(10)
)


# ============================================================
# 10. NLP PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_description"] = (
    df["seller_description"]
    .apply(clean_text)
)

print("\nNLP preprocessing completed!")


# ============================================================
# 11. DISPLAY NLP BEFORE AND AFTER
# ============================================================

print("\nOriginal and processed descriptions:")

print(
    df[["seller_description", "clean_description"]].head(10)
)

# ============================================================
# 12. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "selling_price"
]

for column in numeric_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.extract(
            r"(\d+\.?\d*)"
        )[0]
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Fill any values that became missing
for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

print("\nNumeric columns converted successfully!")


# ============================================================
# 13. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())

plt.figure(figsize=(10, 6))

plt.hist(
    df["selling_price"],
    bins=30
)

plt.title(
    "Selling Price Distribution"
)

plt.xlabel(
    "Selling Price"
)

plt.ylabel(
    "Number of Cars"
)

plt.tight_layout()
plt.show()


# ============================================================
# 15. EDA - VEHICLE AGE VS PRICE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["vehicle_age"],
    df["selling_price"],
    alpha=0.6
)

plt.title(
    "Vehicle Age vs Selling Price"
)

plt.xlabel(
    "Vehicle Age"
)

plt.ylabel(
    "Selling Price"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. EDA - KM DRIVEN VS PRICE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["km_driven"],
    df["selling_price"],
    alpha=0.6
)

plt.title(
    "Kilometers Driven vs Selling Price"
)

plt.xlabel(
    "Kilometers Driven"
)

plt.ylabel(
    "Selling Price"
)

plt.tight_layout()
plt.show()


# ============================================================
# 17. EDA - FUEL TYPE VS PRICE
# ============================================================

fuel_types = (
    df["fuel_type"]
    .dropna()
    .unique()
)

fuel_data = [
    df[
        df["fuel_type"] == fuel
    ]["selling_price"]
    for fuel in fuel_types
]

plt.figure(figsize=(10, 6))

plt.boxplot(
    fuel_data,
    labels=fuel_types
)

plt.title(
    "Fuel Type vs Selling Price"
)

plt.xlabel(
    "Fuel Type"
)

plt.ylabel(
    "Selling Price"
)

plt.tight_layout()
plt.show()


# ============================================================
# 18. CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()
plt.show()


# ============================================================
# 19. DEFINE FEATURES
# ============================================================

numeric_features = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

categorical_features = [
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type"
]

text_feature = "clean_description"

target = "selling_price"

X = df[
    numeric_features
    + categorical_features
    + [text_feature]
]

y = df[target]

print("\n" + "=" * 70)
print("FEATURES")
print("=" * 70)

print(X.columns.tolist())

print("\nTarget:")
print(target)


# ============================================================
# 20. TRAIN-TEST SPLIT - 80:20
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 21. NUMERIC PREPROCESSOR
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 22. CATEGORICAL PREPROCESSOR
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 23. NLP TF-IDF PREPROCESSOR
# ============================================================

text_transformer = TfidfVectorizer(
    max_features=300,
    stop_words="english",
    ngram_range=(1, 2)
)


# ============================================================
# 24. CREATE HYBRID PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),

        (
            "text",
            text_transformer,
            text_feature
        )
    ]
)

print(
    "\nHybrid preprocessing pipeline created!"
)

print(
    "Structured features + Categorical features + TF-IDF text"
)


# ============================================================
# 25. CREATE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=20
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            max_depth=25,
            n_jobs=-1
        )
}

print("\n" + "=" * 70)
print("MODELS")
print("=" * 70)

for model_name in models:
    print("-", model_name)


# ============================================================
# 26. TRAIN ALL MODELS
# ============================================================

trained_models = {}

for model_name, model in models.items():

    print(
        "\nTraining:",
        model_name
    )

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_models[
        model_name
    ] = pipeline

    print(
        "Training completed!"
    )


# ============================================================
# 27. EVALUATE MODELS
# ============================================================

results = []

for model_name, model in trained_models.items():

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2 Score": r2
    })


# ============================================================
# 28. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    results_df.round(4)
)


# ============================================================
# 29. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.title(
    "Model Comparison Based on R² Score"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R² Score"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()
plt.show()


# ============================================================
# 30. SELECT MODEL FOR FINAL PREDICTION
# ============================================================

best_model_name = results_df.loc[
    results_df["R2 Score"].idxmax(),
    "Model"
]

best_model = trained_models[
    best_model_name
]

print(
    "\nSelected model for prediction:",
    best_model_name
)


# ============================================================
# 31. ACTUAL VS PREDICTED PRICE
# ============================================================

predictions = best_model.predict(
    X_test
)

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

plt.xlabel(
    "Actual Selling Price"
)

plt.ylabel(
    "Predicted Selling Price"
)

plt.title(
    "Actual vs Predicted Selling Price"
)

plt.tight_layout()
plt.show()


# ============================================================
# 32. SAMPLE PREDICTIONS
# ============================================================

sample_predictions = pd.DataFrame({

    "Actual Price":
        y_test.values[:10],

    "Predicted Price":
        predictions[:10]
})

sample_predictions[
    "Difference"
] = (
    sample_predictions["Actual Price"]
    -
    sample_predictions["Predicted Price"]
)

print("\nSample predictions:")

print(
    sample_predictions.round(2)
)


# ============================================================
# 33. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    best_model,
    "used_car_price_model.pkl"
)

print(
    "\nTrained model saved as:"
)

print(
    "used_car_price_model.pkl"
)


# ============================================================
# 34. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_results.csv",
    index=False
)

print(
    "\nModel results saved as:"
)

print(
    "model_results.csv"
)


# ============================================================
# 35. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "processed_car_dataset_900.csv",
    index=False
)

print(
    "\nProcessed dataset saved as:"
)

print(
    "processed_car_dataset_900.csv"
)


# ============================================================
# 36. FINAL TEST USING ONE REAL TEST RECORD
# ============================================================

sample_car = X_test.iloc[[0]]

actual_price = y_test.iloc[0]

predicted_price = best_model.predict(
    sample_car
)[0]

print("\n" + "=" * 70)
print("FINAL USED CAR PRICE PREDICTION")
print("=" * 70)

print(
    "Actual Price    : ₹{:,.0f}".format(
        actual_price
    )
)

print(
    "Predicted Price : ₹{:,.0f}".format(
        predicted_price
    )
)

print(
    "Model Used      :",
    best_model_name
)

print("=" * 70)

print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")