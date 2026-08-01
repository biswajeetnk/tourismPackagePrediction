import pandas as pd

# Load dataset
df = pd.read_csv("data/tourism.csv")

# Expected columns
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Validate columns
if list(df.columns) == expected_columns:
    print("Dataset validation successful.")
else:
    raise Exception("Dataset validation failed. Column mismatch.")

# Print summary
print("\nDataset Summary")
print("---------------------------")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print("\nMissing Values")
print(df.isnull().sum())
print("\nData Types")
print(df.dtypes)
