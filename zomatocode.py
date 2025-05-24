import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("zomato.csv.zip", encoding='latin-1')

# Check basic info
print(df.shape)
print(df.columns)
print(df.head())

# Drop unnecessary columns
df = df.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list', 'listed_in(city)'], axis=1, errors='ignore')

# Remove duplicate entries
df.drop_duplicates(inplace=True)

# Drop rows with null values in important columns
df = df.dropna(subset=['rate', 'location', 'cuisines', 'approx_cost(for two people)'])

# Clean 'rate' column (remove '/5', convert to float)
# Remove 'NEW', '-', and nulls from 'rate' and convert to float
# Clean and convert 'rate' safely
def safe_convert_rate(x):
    try:
        if isinstance(x, str):
            if x.strip() in ['NEW', '-', '']:
                return None
            return float(x.split('/')[0].strip())
        return float(x)
    except:
        return None

df['rate'] = df['rate'].apply(safe_convert_rate)
df.dropna(subset=['rate'], inplace=True)

# Clean 'approx_cost(for two people)' safely
df['approx_cost(for two people)'] = (
    df['approx_cost(for two people)']
    .astype(str)
    .str.replace(',', '', regex=False)
)

df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
df.dropna(subset=['approx_cost(for two people)'], inplace=True)

# Rename columns for ease
df.rename(columns={
    'name': 'Restaurant',
    'online_order': 'Online_order',
    'book_table': 'Book_table',
    'rate': 'Rating',
    'approx_cost(for two people)': 'Cost_for_two',
    'listed_in(type)': 'Type',
}, inplace=True)

# ---------------------------
# 📊 VISUALIZATIONS
# ---------------------------

sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))

# 1. Cities with most restaurants
top_cities = df['location'].value_counts().head(10)
sns.barplot(x=top_cities.values, y=top_cities.index, palette='viridis')
plt.title('Top 10 Locations with Most Restaurants')
plt.xlabel('Number of Restaurants')
plt.ylabel('Location')
plt.tight_layout()
plt.show()

# 2. Most popular cuisines
top_cuisines = df['cuisines'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='magma')
plt.title('Top 10 Cuisines')
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisine')
plt.tight_layout()
plt.show()

# 3. Ratings Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=20, kde=True, color='teal')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Online Order Availability
plt.figure(figsize=(6, 4))
sns.countplot(x='Online_order', data=df, palette='pastel')
plt.title('Online Order Availability')
plt.tight_layout()
plt.show()

# 5. Cost for Two Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Cost_for_two'], bins=30, kde=True, color='coral')
plt.title('Cost for Two People Distribution')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 6. Average Rating per Restaurant Type
plt.figure(figsize=(12, 6))
type_rating = df.groupby('Type')['Rating'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=type_rating.values, y=type_rating.index, palette='coolwarm')
plt.title('Average Rating by Restaurant Type')
plt.xlabel('Average Rating')
plt.ylabel('Type')
plt.tight_layout()
plt.show()
