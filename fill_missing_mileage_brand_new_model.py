import pandas as pd

# 1. Load the dataset
file_path = 'dataset/riyasewana_toyota_yaris_with_country.csv'
df = pd.read_csv(file_path)

# 2. Fill missing values in the 'Mileage' column with '0 km'
df['Mileage'] = df['Mileage'].fillna('0 km')

# (Optional) If you also need to fill missing 'Price' values with 0
# df['Price'] = df['Price'].fillna('0')

# 3. Save the updated dataframe to a new CSV file
output_path = 'dataset/riyasewana_toyota_yaris.csv'
df.to_csv(output_path, index=False)

print(f"Missing values filled and saved successfully to {output_path}!")