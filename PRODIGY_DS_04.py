import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#-------------------------------------------------------------------------------------

# Reload the dataset with the correct delimiter
twitter_data = pd.read_csv(r"C:\Users\fatem\Desktop\Prodigy\Task 04\archive (3)\twitter_training.csv")

# Display the first few rows to verify correct parsing
print(twitter_data.head())

# Display column-wise missing values
missing_values = twitter_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Check for duplicates
duplicate_rows = twitter_data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}\n")

# Display a sample of the data
print("\nSample Data:")
display(twitter_data.head())

#-------------------------------------------------------------------------------------------------------

# Preprocessing
# Convert all text to strings and handle non-string values
def preprocess_text(text):
    if isinstance(text, str):  # Check if the text is a string
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""  # Return an empty string for non-string values

# Apply preprocessing to the text column
twitter_data['Processed_Text'] = twitter_data['im getting on borderlands and i will murder you all ,'].apply(preprocess_text)

# Display cleaned data
twitter_data_info = twitter_data.info()
twitter_data_sample = twitter_data[['Positive', 'im getting on borderlands and i will murder you all ,', 'Processed_Text']].sample(5, random_state=42)

(twitter_data_info, twitter_data_sample)

#-----------------------------------------------------------------------------------------------------------

# EDA
# Visualize sentiment distribution
# Count sentiments
sentiment_counts = twitter_data['Positive'].value_counts()

# Plot the distribution
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# Sentiment distribution by topic
sentiment_by_topic = twitter_data.groupby(['Borderlands', 'Positive']).size().unstack(fill_value=0)

# Visualize sentiment distribution for top topics
top_topics = sentiment_by_topic.sum(axis=1).sort_values(ascending=False).head(5).index
filtered_sentiment_by_topic = sentiment_by_topic.loc[top_topics]

# Plot
filtered_sentiment_by_topic.plot(kind='bar', figsize=(10, 6), stacked=True)
plt.title('Sentiment Distribution by Top Topics')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()


# Generate word clouds for each sentiment
sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
for sentiment in sentiments:
    text = ' '.join(twitter_data[twitter_data['Positive'] == sentiment]['Processed_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Frequent Words in {sentiment} Sentiment')
    plt.show()
