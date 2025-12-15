# %%
# ============================================
# CELL 1: Imports and Setup
# ============================================

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
def load_jsonl(file_path, sample_size=None):
    """
    Load data from .jsonl file
    
    Args:
        file_path: Path to your .jsonl file
        sample_size: Number of lines to load (None = all)
    
    Returns:
        DataFrame with episode data
    """
    print(f"Loading data from {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {i}: {e}")
                continue
    
    df = pd.DataFrame(data)
    print(f" Loaded {len(df)} episodes")
    print(f" Columns: {df.columns.tolist()}")
    
    return df

file_path = 'episodeLevelDataSample.jsonl'  # Change to your actual file path
sample_size = 1000  # Start with 1000 episodes for testing

# Load data
episodes_df = load_jsonl(file_path)

# Display first few rows
print("\nFirst 3 rows:")
episodes_df.head(3)



# %%
# ============================================
# CELL 3: Basic Data Overview
# ============================================


print(f"\nDataset shape: {episodes_df.shape}")
print(f"Number of episodes: {len(episodes_df)}")
print(f"Number of columns: {len(episodes_df.columns)}")

print("\nColumn names and types:")
print(episodes_df.dtypes)

# %%
print("\nMissing values:")
missing = episodes_df.isnull().sum()
missing_pct = (missing / len(episodes_df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

# %%
# ============================================
# CELL 5: Text Content Analysis
# ============================================

transcript_col = None
for col in ['transcript', 'text', 'content', 'description', 'show_description']:
    if col in episodes_df.columns:
        transcript_col = col
        break

if transcript_col:
    print(f"\n Found transcript column: '{transcript_col}'")
    
    # Calculate text lengths
    episodes_df['text_length'] = episodes_df[transcript_col].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    episodes_df['word_count'] = episodes_df[transcript_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    print("\nText length statistics (characters):")
    print(episodes_df['text_length'].describe())
    
    print("\nWord count statistics:")
    print(episodes_df['word_count'].describe())
    
    # Show sample transcript
    print("\n" + "="*60)
    print("SAMPLE TRANSCRIPT")
    print("="*60)
    idx = episodes_df['text_length'].idxmax()  # Longest transcript
    print(f"\nEpisode with longest transcript (index {idx}):")
    print(str(episodes_df.loc[idx, transcript_col])[:500] + "...")

# %%
# ============================================
# CELL 6: Categorical Analysis - Shows
# ============================================

print("="*60)
print("PODCAST SHOWS ANALYSIS")
print("="*60)

# Find show name column
show_col = None
for col in ['podTitle', 'show', 'podcast_name', 'show_filename_prefix']:
    if col in episodes_df.columns:
        show_col = col
        break

if show_col:
    print(f"\n Found show column: '{show_col}'")
    
    show_counts = episodes_df[show_col].value_counts()
    print(f"\nNumber of unique shows: {len(show_counts)}")
    print(f"\nTop 10 shows with most episodes:")
    print(show_counts.head(10))
    
    # Statistics
    print(f"\nEpisodes per show statistics:")
    print(f"  Mean: {show_counts.mean():.2f}")
    print(f"  Median: {show_counts.median():.2f}")
    print(f"  Max: {show_counts.max()}")
    print(f"  Min: {show_counts.min()}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    show_counts.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Top 15 Podcast Shows by Number of Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Show Name', fontsize=12)
    plt.ylabel('Number of Episodes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    

# %%
# ============================================
# CELL 7: Categorical Analysis - Categories
# ============================================

# Find category column
category_col = None
for col in ['category1', 'categories', 'genre', 'rss_category']:
    if col in episodes_df.columns:
        category_col = col
        break

if category_col:
    print(f"\n Found category column: '{category_col}'")
    
    # Handle if categories is a list
    if episodes_df[category_col].dtype == 'object':
        # Check if first value is a list
        first_val = episodes_df[category_col].iloc[0]
        if isinstance(first_val, list):
            # Flatten list of categories
            all_categories = []
            for cats in episodes_df[category_col]:
                if isinstance(cats, list):
                    all_categories.extend(cats)
                else:
                    all_categories.append(cats)
            category_counts = pd.Series(all_categories).value_counts()
            print("(Note: Episodes can have multiple categories)")
        else:
            category_counts = episodes_df[category_col].value_counts()
    else:
        category_counts = episodes_df[category_col].value_counts()
    
    print(f"\nNumber of unique categories: {len(category_counts)}")
    print(f"\nTop 15 categories:")
    print(category_counts.head(15))
    
    # Visualize
    plt.figure(figsize=(12, 6))
    category_counts.head(15).plot(kind='barh', color='coral', edgecolor='black')
    plt.title('Top 15 Podcast Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Episodes', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.tight_layout()
    plt.show()
    

# %%
# ============================================
# CELL 8: Text Length Distribution
# ============================================

if transcript_col and 'text_length' in episodes_df.columns:
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Character length histogram
    axes[0, 0].hist(episodes_df['text_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Transcript Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Transcript Lengths')
    axes[0, 0].axvline(episodes_df['text_length'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {episodes_df['text_length'].mean():.0f}")
    axes[0, 0].legend()
    
    # Word count histogram
    axes[0, 1].hist(episodes_df['word_count'], bins=50, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Word Counts')
    axes[0, 1].axvline(episodes_df['word_count'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {episodes_df['word_count'].mean():.0f}")
    axes[0, 1].legend()
    
    # Box plot - character length
    axes[1, 0].boxplot(episodes_df['text_length'], vert=True)
    axes[1, 0].set_ylabel('Transcript Length (characters)')
    axes[1, 0].set_title('Box Plot: Transcript Lengths')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot - word count
    axes[1, 1].boxplot(episodes_df['word_count'], vert=True)
    axes[1, 1].set_ylabel('Word Count')
    axes[1, 1].set_title('Box Plot: Word Counts')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Identify potential issues
    print("\nPotential data quality issues:")
    
    very_short = (episodes_df['word_count'] < 50).sum()
    print(f"  - Episodes with <50 words: {very_short} ({very_short/len(episodes_df)*100:.1f}%)")
    
    very_long = (episodes_df['word_count'] > 50000).sum()
    print(f"  - Episodes with >50K words: {very_long} ({very_long/len(episodes_df)*100:.1f}%)")
    
    empty = (episodes_df['text_length'] == 0).sum()
    print(f"  - Empty transcripts: {empty} ({empty/len(episodes_df)*100:.1f}%)")

# %%
# ============================================
# CELL 9: Temporal Analysis (if date available)
# ============================================

date_col = None
for col in ['date', 'publish_date', 'published_date', 'publication_date', 'episode_pub_date']:
    if col in episodes_df.columns:
        date_col = col
        break

if date_col:
    print(f"\n Found date column: '{date_col}'")
    
    # Convert to datetime
    episodes_df['pub_date'] = pd.to_datetime(episodes_df[date_col], errors='coerce')
    
    # Remove rows where date conversion failed
    valid_dates = episodes_df['pub_date'].notna()
    print(f"\nEpisodes with valid dates: {valid_dates.sum()} ({valid_dates.sum()/len(episodes_df)*100:.1f}%)")
    
    if valid_dates.sum() > 0:
        date_df = episodes_df[valid_dates].copy()
        
        print(f"\nDate range:")
        print(f"  Earliest: {date_df['pub_date'].min()}")
        print(f"  Latest: {date_df['pub_date'].max()}")
        
        # Extract month and year
        date_df['year_month'] = date_df['pub_date'].dt.to_period('M')
        
        # Count by month
        monthly_counts = date_df['year_month'].value_counts().sort_index()
        
        # Plot timeline
        plt.figure(figsize=(14, 6))
        monthly_counts.plot(kind='line', marker='o', color='purple', linewidth=2)
        plt.title('Episodes Published Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Episodes', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
else:
    print("\n No date column found")


# %%
# ============================================
# CELL 10: Word Cloud (Top Keywords)
# ============================================

if transcript_col:
    print("="*60)
    print("KEYWORD ANALYSIS")
    print("="*60)
    
    # Combine all transcripts
    all_text = ' '.join(episodes_df[transcript_col].dropna().astype(str))
    
    # Simple word frequency (without proper NLP preprocessing)
    from collections import Counter
    import re
    
    # Basic tokenization
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'for', 'this', 'that', 'with', 'from', 'have', 
                 'but', 'not', 'they', 'was', 'are', 'been', 'will', 'can', 
                 'just', 'about', 'like', 'know', 'think', 'get', 'going', 
                 'said', 'one', 'would', 'could', 'its', 'more', 'when', 
                 'what', 'there', 'out', 'all', 'were', 'had', 'has'}
    
    words = [w for w in words if w not in stopwords]
    
    # Count word frequencies
    word_counts = Counter(words)
    top_words = word_counts.most_common(30)
    
    print("\nTop 30 most frequent words:")
    for word, count in top_words:
        print(f"  {word}: {count}")
    
    # Visualize top words
    words_df = pd.DataFrame(top_words[:20], columns=['Word', 'Frequency'])
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(words_df)), words_df['Frequency'], color='teal', edgecolor='black')
    plt.yticks(range(len(words_df)), words_df['Word'])
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top 20 Most Frequent Words in Transcripts', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# %%
# ============================================
# CELL 11: Data Quality Summary
# ============================================

quality_issues = []

# Check transcript quality
if transcript_col:
    empty_transcripts = episodes_df[transcript_col].isna().sum()
    if empty_transcripts > 0:
        quality_issues.append(f" {empty_transcripts} episodes with missing transcripts")
    
    if 'word_count' in episodes_df.columns:
        short_transcripts = (episodes_df['word_count'] < 50).sum()
        if short_transcripts > 0:
            quality_issues.append(f" {short_transcripts} episodes with <50 words")

# Check metadata quality
if show_col:
    missing_show = episodes_df[show_col].isna().sum()
    if missing_show > 0:
        quality_issues.append(f" {missing_show} episodes without show name")

if category_col:
    missing_category = episodes_df[category_col].isna().sum()
    if missing_category > 0:
        quality_issues.append(f" {missing_category} episodes without category")

# Print summary
if quality_issues:
    print("\nData quality issues found:")
    for issue in quality_issues:
        print(f"  {issue}")

# %%
# ============================================
# CELL 12: Save Column Mapping for Next Steps
# ============================================


# Create a mapping of standard names to actual column names
column_mapping = {
    'transcript': transcript_col,
    'show_name': show_col,
    'category': category_col,
}

print("\nColumn mapping identified:")
for standard_name, actual_name in column_mapping.items():
    if actual_name:
        print(f"  {standard_name} → '{actual_name}'")
    else:
        print(f"  {standard_name} → NOT FOUND")


import json
with open('column_mapping.json', 'w') as f:
    json.dump(column_mapping, f, indent=2)

print("\n Saved column mapping to 'column_mapping.json'")

# %%
# ============================================
# CELL 13: Display Summary Statistics
# ============================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary_stats = {
    'Total Episodes': len(episodes_df),
    'Unique Shows': episodes_df[show_col].nunique() if show_col else 'N/A',
    'Unique Categories': episodes_df[category_col].nunique() if category_col else 'N/A',
    'Avg Words per Episode': f"{episodes_df['word_count'].mean():.0f}" if 'word_count' in episodes_df.columns else 'N/A',
    'Median Words per Episode': f"{episodes_df['word_count'].median():.0f}" if 'word_count' in episodes_df.columns else 'N/A',
    'Date Range': f"{episodes_df['pub_date'].min()} to {episodes_df['pub_date'].max()}" if 'pub_date' in episodes_df.columns else 'N/A'
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
print(summary_df.to_string(index=False))


# %% [markdown]
# ### Preprocessing

# %%
# Load your data
file_path = 'episodeLevelDataSample.jsonl'
sample_size = 10000

print(f"Loading {sample_size} episodes from {file_path}...")

data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= sample_size:
            break
        try:
            data.append(json.loads(line))
        except:
            continue

episodes_df = pd.DataFrame(data)
print(f"✓ Loaded {len(episodes_df)} episodes")


# %%
print("\nKey columns we'll use:")
print("  Transcript: 'transcript'")
print("  Show: 'podTitle'")
print("  Category: 'category1' (primary)")
print("  Episode Title: 'epTitle'")
print("  Date: 'episodeDateLocalized'")

# %%
# ============================================
# CELL 2: Handle Categories (Multiple Category Columns)
# ============================================

# dataset has category1-10 (hierarchical categories)
# We'll create a primary category and a list of all categories

def extract_categories(row):
    """Extract all non-null categories for an episode"""
    cats = []
    for i in range(1, 11):
        col = f'category{i}'
        if col in row and pd.notna(row[col]):
            cats.append(row[col])
    return cats


# %%
# Create category fields
episodes_df['primary_category'] = episodes_df['category1'].fillna('Unknown')
episodes_df['all_categories'] = episodes_df.apply(extract_categories, axis=1)
episodes_df['num_categories'] = episodes_df['all_categories'].apply(len)

print(f"\nCategory statistics:")
print(f"  Episodes with primary category: {episodes_df['primary_category'].notna().sum()}")
print(f"  Average categories per episode: {episodes_df['num_categories'].mean():.2f}")

print(f"\nTop 10 primary categories:")
print(episodes_df['primary_category'].value_counts().head(10))

# %%
# Create a simplified category (for cleaner grouping)
# Map similar categories together
def simplify_category(cat):
    """Simplify category names for better grouping"""
    if pd.isna(cat) or cat == 'Unknown':
        return 'Unknown'
    
    cat_lower = cat.lower()
    
    # Map to broader categories
    if any(x in cat_lower for x in ['news', 'politics', 'government']):
        return 'News & Politics'
    elif any(x in cat_lower for x in ['business', 'entrepreneur', 'marketing', 'investing']):
        return 'Business'
    elif any(x in cat_lower for x in ['comedy', 'humor', 'improv']):
        return 'Comedy'
    elif any(x in cat_lower for x in ['education', 'learning', 'courses']):
        return 'Education'
    elif any(x in cat_lower for x in ['health', 'fitness', 'medicine', 'mental']):
        return 'Health & Wellness'
    elif any(x in cat_lower for x in ['tech', 'technology', 'science']):
        return 'Technology & Science'
    elif any(x in cat_lower for x in ['society', 'culture', 'documentary']):
        return 'Society & Culture'
    elif any(x in cat_lower for x in ['sport', 'athletics']):
        return 'Sports'
    elif any(x in cat_lower for x in ['music', 'arts']):
        return 'Arts & Music'
    elif any(x in cat_lower for x in ['religion', 'spirituality']):
        return 'Religion & Spirituality'
    elif any(x in cat_lower for x in ['true crime', 'crime']):
        return 'True Crime'
    else:
        return cat  # Keep original if no match

episodes_df['category_simplified'] = episodes_df['primary_category'].apply(simplify_category)

print(f"\nSimplified categories:")
print(episodes_df['category_simplified'].value_counts().head(10))

# %%
# ============================================
# CELL 3: Clean Transcripts
# ============================================


def clean_transcript(text):
    """
    Clean transcript text for modeling
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

episodes_df['transcript_clean'] = episodes_df['transcript'].apply(clean_transcript)

# Calculate statistics
episodes_df['word_count'] = episodes_df['transcript_clean'].apply(lambda x: len(x.split()))
episodes_df['char_count'] = episodes_df['transcript_clean'].apply(len)

print(f"\nWord count statistics:")
print(episodes_df['word_count'].describe())

# %%
# ============================================
# CELL 4: Filter Low-Quality Episodes
# ============================================

initial_count = len(episodes_df)

# Filter criteria
min_words = 100  # At least 100 words
max_words = 80000  # Remove extremely long transcripts (potential errors)

# Apply filters
episodes_df = episodes_df[
    (episodes_df['word_count'] >= min_words) & 
    (episodes_df['word_count'] <= max_words)
].copy()

removed = initial_count - len(episodes_df)

print(f"Initial episodes: {initial_count}")
print(f"Removed: {removed} episodes")
print(f"  - Too short (<{min_words} words): {(initial_count - len(episodes_df[episodes_df['word_count'] >= min_words]))}")
print(f"Final dataset: {len(episodes_df)} episodes")

# Reset index
episodes_df = episodes_df.reset_index(drop=True)


# %%
# ============================================
# CELL 5: Handle Show Names
# ============================================

# Clean show names
episodes_df['show_name'] = episodes_df['podTitle'].fillna('Unknown')

print(f"\nTop 10 shows by episode count:")
print(episodes_df['show_name'].value_counts().head(10))

# Identify shows with multiple episodes (useful for evaluation)
show_counts = episodes_df['show_name'].value_counts()
multi_episode_shows = show_counts[show_counts > 1]

print(f"\nShows with multiple episodes: {len(multi_episode_shows)}")
print(f"Episodes from multi-episode shows: {multi_episode_shows.sum()}")

# %%
# ============================================
# CELL 6: Process Dates
# ============================================

# Convert timestamp to datetime
episodes_df['pub_date'] = pd.to_datetime(episodes_df['episodeDateLocalized'], unit='s', errors='coerce')

valid_dates = episodes_df['pub_date'].notna()
print(f"Episodes with valid dates: {valid_dates.sum()} ({valid_dates.sum()/len(episodes_df)*100:.1f}%)")

if valid_dates.sum() > 0:
    print(f"\nDate range:")
    print(f"  Earliest: {episodes_df['pub_date'].min()}")
    print(f"  Latest: {episodes_df['pub_date'].max()}")
    
    # Extract temporal features
    episodes_df['year'] = episodes_df['pub_date'].dt.year
    episodes_df['month'] = episodes_df['pub_date'].dt.month
    episodes_df['year_month'] = episodes_df['pub_date'].dt.to_period('M')

# %%
# ============================================
# CELL 7: Process Duration
# ============================================

# Duration in seconds
episodes_df['duration_minutes'] = episodes_df['durationSeconds'] / 60

print(f"Duration statistics (minutes):")
print(episodes_df['duration_minutes'].describe())

# Categorize by length
def categorize_duration(minutes):
    if pd.isna(minutes):
        return 'Unknown'
    elif minutes < 15:
        return 'Short (<15min)'
    elif minutes < 45:
        return 'Medium (15-45min)'
    elif minutes < 90:
        return 'Long (45-90min)'
    else:
        return 'Very Long (>90min)'

episodes_df['duration_category'] = episodes_df['duration_minutes'].apply(categorize_duration)

print(f"\nDuration distribution:")
print(episodes_df['duration_category'].value_counts())

# %%
# ============================================
# CELL 8: Process Speaker Information
# ============================================

# Number of hosts/guests
print(f"Host statistics:")
print(episodes_df['numUniqueHosts'].describe())

print(f"\nGuest statistics:")
print(episodes_df['numUniqueGuests'].describe())

# Total speakers
episodes_df['total_speakers'] = episodes_df['numUniqueHosts'].fillna(0) + episodes_df['numUniqueGuests'].fillna(0)

print(f"\nTotal speakers per episode:")
print(episodes_df['total_speakers'].describe())

# Categorize by speaker count
def categorize_speakers(count):
    if pd.isna(count) or count == 0:
        return 'Unknown'
    elif count == 1:
        return 'Solo'
    elif count == 2:
        return 'Duo'
    else:
        return 'Multiple (3+)'

episodes_df['speaker_category'] = episodes_df['total_speakers'].apply(categorize_speakers)

print(f"\nSpeaker distribution:")
print(episodes_df['speaker_category'].value_counts())

# %%
# ============================================
# CELL 9: Create Final Feature Set
# ============================================

# Select key columns for modeling
key_columns = [
    # Core content
    'transcript_clean',
    'epTitle',
    
    # Metadata
    'show_name',
    'primary_category',
    'category_simplified',
    'all_categories',
    
    # Episode characteristics
    'word_count',
    'char_count',
    'duration_minutes',
    'duration_category',
    
    # Speaker info
    'numUniqueHosts',
    'numUniqueGuests',
    'total_speakers',
    'speaker_category',
    
    # Original columns (for reference)
    'epDescription',
    'podDescription'
]

# Keep only available columns
available_columns = [col for col in key_columns if col in episodes_df.columns]
modeling_df = episodes_df[available_columns].copy()

print(f"Final dataset shape: {modeling_df.shape}")
print(f"Columns selected: {len(available_columns)}")

print("\nFinal column list:")
for col in available_columns:
    print(f"  - {col}")

# %%
# ============================================
# CELL 10: Data Quality Report
# ============================================

print("\n" + "="*60)
print("FINAL DATA QUALITY REPORT")
print("="*60)

quality_report = {
    'Total Episodes': len(modeling_df),
    'Unique Shows': modeling_df['show_name'].nunique(),
    'Unique Categories': modeling_df['category_simplified'].nunique(),
    'Avg Words per Episode': f"{modeling_df['word_count'].mean():.0f}",
    'Median Words per Episode': f"{modeling_df['word_count'].median():.0f}",
    'Avg Duration (min)': f"{modeling_df['duration_minutes'].mean():.1f}",
    'Date Range': f"{modeling_df['pub_date'].min()} to {modeling_df['pub_date'].max()}" if 'pub_date' in modeling_df.columns else 'N/A'
}

for metric, value in quality_report.items():
    print(f"  {metric}: {value}")

# Missing value check
print("\nMissing values in key columns:")
missing = modeling_df[['transcript_clean', 'show_name', 'category_simplified']].isnull().sum()
print(missing)


# %%
# ============================================
# CELL 11: Save Processed Data
# ============================================

# Save to pickle (preserves data types)
modeling_df.to_pickle('processed_episodes.pkl')
print(" Saved to 'data/processed_episodes.pkl'")

# Also save as CSV for inspection
modeling_df.to_csv('processed_episodes.csv', index=False)
print(" Saved to 'data/processed_episodes.csv'")

# Save column mapping
column_mapping = {
    'transcript': 'transcript_clean',
    'show_name': 'show_name',
    'category': 'category_simplified',
    'primary_category': 'primary_category',
    'duration': 'duration_minutes',
    'title': 'epTitle'
}

with open('column_mapping.json', 'w') as f:
    json.dump(column_mapping, f, indent=2)

print(" Saved column mapping to 'column_mapping.json'")

# %%
# ============================================
# CELL 13: Summary Statistics by Category
# ============================================

print("\n" + "="*60)
print("STATISTICS BY CATEGORY")
print("="*60)

category_stats = modeling_df.groupby('category_simplified').agg({
    'word_count': ['mean', 'median'],
    'duration_minutes': ['mean', 'median'],
    'show_name': 'nunique',
    'transcript_clean': 'count'
}).round(2)

category_stats.columns = ['Avg Words', 'Median Words', 'Avg Duration', 'Median Duration', 'Unique Shows', 'Episode Count']
category_stats = category_stats.sort_values('Episode Count', ascending=False)

print("\nTop categories:")
print(category_stats.head(10))




