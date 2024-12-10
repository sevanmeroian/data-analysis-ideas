import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import os

# Define output directory
output_directory = r'C:\Users\flipp\source\repos\Python Script for Reduced Data\Error_Charts'
os.makedirs(output_directory, exist_ok=True)

# Load the datasets
true_data_path = r'C:\Users\flipp\source\repos\Python Script for Reduced Data\True_Reduced(in).csv'
fake_data_path = r'C:\Users\flipp\source\repos\Python Script for Reduced Data\Fake_Reduced(in).csv'

try:
    true_data = pd.read_csv(true_data_path, low_memory=False)
    fake_data = pd.read_csv(fake_data_path, low_memory=False)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Add labels and combine datasets
true_data['label'] = 'True'
fake_data['label'] = 'Fake'
combined_data = pd.concat([true_data, fake_data], ignore_index=True)

# Handle date column
if 'date' in combined_data.columns:
    try:
        combined_data['date'] = pd.to_datetime(combined_data['date'], format='%Y-%m-%d', errors='coerce')  # Specify format
    except Exception as e:
        print(f"Error parsing dates: {e}")
else:
    print("Warning: 'date' column not found in the dataset.")

# Check date parsing issues
invalid_dates = combined_data[combined_data['date'].isna()].shape[0]
future_dates = combined_data[combined_data['date'] > pd.Timestamp.now()].shape[0]
valid_dates = len(combined_data) - invalid_dates - future_dates

# 1. Duplicates
duplicates = combined_data.duplicated(subset=['title', 'text']).sum()
duplicate_data = pd.Series({'Duplicates': duplicates, 'Unique Rows': len(combined_data) - duplicates})
duplicate_data.plot(kind='pie', autopct='%1.1f%%', startangle=90, figsize=(8, 8), title='Duplicate Rows Breakdown')
plt.ylabel('')  # Remove y-axis label
plt.savefig(os.path.join(output_directory, 'duplicate_rows_breakdown.png'))
plt.close()

# 2. Date Validity Breakdown
date_data = pd.Series({'Valid Dates': valid_dates, 'Invalid Dates': invalid_dates, 'Future Dates': future_dates})
date_data.plot(kind='pie', autopct='%1.1f%%', startangle=90, figsize=(8, 8), title='Date Validity Breakdown')
plt.ylabel('')
plt.savefig(os.path.join(output_directory, 'date_validity_breakdown.png'))
plt.close()

# 3. Sentiment vs Label
combined_data['sentiment'] = combined_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
combined_data['sentiment_category'] = combined_data['sentiment'].apply(
    lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
)
sentiment_vs_label = combined_data.groupby(['label', 'sentiment_category']).size().unstack(fill_value=0)
sentiment_vs_label.plot(kind='bar', stacked=True, figsize=(10, 6), title='Sentiment vs. Label Breakdown')
plt.xlabel('Label')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.savefig(os.path.join(output_directory, 'sentiment_vs_label_breakdown.png'))
plt.close()

# 4. Missing Values
missing_values = combined_data.isnull().sum()
missing_values = missing_values[missing_values > 0]  # Only include columns with missing values
if not missing_values.empty:
    missing_values.plot(kind='bar', figsize=(10, 6), title='Missing Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('Count of Missing Values')
    plt.savefig(os.path.join(output_directory, 'missing_values_breakdown.png'))
    plt.close()

print(f"Charts saved to: {output_directory}")
