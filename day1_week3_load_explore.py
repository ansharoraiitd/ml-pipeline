# day1_week3_load_explore.py
from data_loader import load_adult_income, real_data_sanity_check

df = load_adult_income()
real_data_sanity_check(df)

print("\n" + "=" * 60)
print("REDUNDANCY CHECK: education vs education-num")
print("=" * 60)
# If these two columns encode the same information, a crosstab
# should show each education-num value maps to exactly one
# education label (and vice versa) — confirm rather than assume
redundancy_check = df.groupby("education-num")["education"].nunique()
print(redundancy_check)
print(f"\nEach education-num maps to exactly 1 education label: "
      f"{(redundancy_check == 1).all()}")

print("\n" + "=" * 60)
print("FIRST LOOK AT NUMERICAL FEATURES")
print("=" * 60)
print(df.describe())

df.to_csv("adult_income_raw.csv", index=False)
print("\nSaved adult_income_raw.csv — the untouched raw data, "
      "kept as a reference point before any cleaning happens")