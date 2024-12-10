import pandas as pd
import matplotlib.pyplot as plt
import os

# Define output directory for saving images
output_directory = r'C:\Users\flipp\source\repos\Python Script for Reduced Data'
os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists

# Load the datasets
true_data_path = r'C:\Users\flipp\source\repos\Python Script for Reduced Data\True_Reduced(in).csv'
fake_data_path = r'C:\Users\flipp\source\repos\Python Script for Reduced Data\Fake_Reduced(in).csv'

true_data = pd.read_csv(true_data_path, low_memory=False)
fake_data = pd.read_csv(fake_data_path, low_memory=False)

# Add labels and combine the datasets
true_data['label'] = 'True'
fake_data['label'] = 'Fake'
combined_data = pd.concat([true_data, fake_data], ignore_index=True)

# Normalize text (optional for word-based analysis)
combined_data['text'] = combined_data['text'].fillna('').str.lower().str.replace(r'[^\w\s]', '', regex=True)

# 1. Word Count Distribution
combined_data['word_count'] = combined_data['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
combined_data[combined_data['label'] == 'True']['word_count'].plot(kind='hist', bins=50, alpha=0.7, label='True', color='blue')
combined_data[combined_data['label'] == 'Fake']['word_count'].plot(kind='hist', bins=50, alpha=0.7, label='Fake', color='red')
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(output_directory, 'word_count_distribution.png'))  # Save the image
plt.show()

# 2. Title Length Distribution
combined_data['title_length'] = combined_data['title'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
combined_data[combined_data['label'] == 'True']['title_length'].plot(kind='hist', bins=50, alpha=0.7, label='True', color='blue')
combined_data[combined_data['label'] == 'Fake']['title_length'].plot(kind='hist', bins=50, alpha=0.7, label='Fake', color='red')
plt.title('Title Length Distribution')
plt.xlabel('Title Length (Words)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(output_directory, 'title_length_distribution.png'))  # Save the image
plt.show()

# 3. Subject Breakdown
if 'subject' in combined_data.columns:
    subject_counts = combined_data.groupby(['label', 'subject']).size().unstack(fill_value=0)
    subject_counts.plot(kind='bar', figsize=(12, 6), stacked=True)
    plt.title('Subject Distribution by Label')
    plt.ylabel('Article Count')
    plt.xlabel('Subject')
    plt.savefig(os.path.join(output_directory, 'subject_distribution.png'))  # Save the image
    plt.show()

# 4. Sentiment Scores (optional if sentiment analysis is available)
try:
    from textblob import TextBlob

    combined_data['sentiment'] = combined_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    plt.figure(figsize=(10, 6))
    combined_data[combined_data['label'] == 'True']['sentiment'].plot(kind='hist', bins=50, alpha=0.7, label='True', color='blue')
    combined_data[combined_data['label'] == 'Fake']['sentiment'].plot(kind='hist', bins=50, alpha=0.7, label='Fake', color='red')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'sentiment_distribution.png'))  # Save the image
    plt.show()
except ImportError:
    print("TextBlob is not installed. Skipping sentiment analysis visualization.")

# 5. Publication Date Trends (if a 'date' column exists)
if 'date' in combined_data.columns:
    combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')
    combined_data = combined_data.dropna(subset=['date'])  # Drop invalid dates
    date_trends = combined_data.groupby([combined_data['date'].dt.to_period('M'), 'label']).size().unstack(fill_value=0)
    date_trends.plot(figsize=(12, 6))
    plt.title('Publication Trends Over Time')
    plt.ylabel('Number of Articles')
    plt.xlabel('Date')
    plt.savefig(os.path.join(output_directory, 'publication_trends.png'))  # Save the image
    plt.show()
