import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')  # Assuming you have a CSV file named 'data.csv'
print("first 5 rows of the dataset:")
print(df.head())

# understanding the dataset

print("\nDataset Information:")
print(df.shape)

print("\nColumns in the dataset:")
print(df.columns)

print("\nstatistical summary of the dataset:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)

# For numeric columns → fill missing values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


# For text/categorical columns → fill missing values
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Example: filter rows where first numeric column is greater than its median

if len(numeric_columns) > 0:
    column = numeric_columns[0]

    filtered_data = df[df[column] > df[column].median()]

    print("\nFiltered data:")
    print(filtered_data.head())


# ==========================================
# 8. Create a new column
# ==========================================

# Example using the first two numeric columns

if len(numeric_columns) >= 2:

    col1 = numeric_columns[0]
    col2 = numeric_columns[1]

    df["New_Column"] = df[col1] + df[col2]

    print("\nNew column created:")
    print(df[[col1, col2, "New_Column"]].head())


# ==========================================
# 9. Sorting data
# ==========================================

if len(numeric_columns) > 0:

    column = numeric_columns[0]

    sorted_df = df.sort_values(by=column, ascending=False)

    print("\nSorted data:")
    print(sorted_df.head())


# ==========================================
# 10. Pandas GroupBy
# ==========================================

if len(categorical_columns) > 0 and len(numeric_columns) > 0:

    category = categorical_columns[0]
    value = numeric_columns[0]

    grouped_data = df.groupby(category)[value].sum()

    print("\nGrouped data:")
    print(grouped_data)


# ==========================================
# 11. Basic Matplotlib visualization
# ==========================================

if len(numeric_columns) > 0:

    column = numeric_columns[0]

    plt.figure(figsize=(8, 5))

    plt.hist(df[column], bins=20)

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.show()


# ==========================================
# 12. Seaborn visualization
# ==========================================

if len(numeric_columns) >= 2:

    x = numeric_columns[0]
    y = numeric_columns[1]

    plt.figure(figsize=(8, 5))

    sns.scatterplot(data=df, x=x, y=y)

    plt.title(f"{x} vs {y}")

    plt.show()


# ==========================================
# 13. Correlation heatmap
# ==========================================

if len(numeric_columns) >= 2:

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.show()


# ==========================================
# 14. Save cleaned dataset
# ==========================================

df.to_csv("cleaned_data.csv", index=False)

print("\n✅ Data cleaning completed!")
print("✅ Cleaned dataset saved as 'cleaned_data.csv'")