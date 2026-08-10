# day2_week3_preprocessing.py
from sklearn.model_selection import train_test_split
from data_loader import load_adult_income
from preprocessing import clean_columns, handle_missing, encode_features

df = load_adult_income()

print("=" * 60)
print("BEFORE CLEANING")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

df = clean_columns(df)
df = handle_missing(df)

print("\n" + "=" * 60)
print("AFTER DROPPING REDUNDANT/NON-INFORMATIVE COLUMNS + RELABELING MISSING")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nAny remaining '?' values: {(df == '?').sum().sum()}")

# Split BEFORE any fitting happens — the leakage-safe order
X = df.drop(columns=["income"])
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

print("\n" + "=" * 60)
print("ENCODING (fit on train only)")
print("=" * 60)
X_train_encoded, X_test_encoded = encode_features(X_train, X_test)

print(f"\nEncoded train shape: {X_train_encoded.shape}")
print(f"Encoded test shape: {X_test_encoded.shape}")
print(f"\nSample encoded columns: {list(X_train_encoded.columns[:10])}")

X_train_encoded.to_csv("X_train_encoded.csv", index=False)
X_test_encoded.to_csv("X_test_encoded.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("\nSaved encoded train/test splits for tomorrow's pipeline work")