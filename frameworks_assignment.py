# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from collections import Counter
import re
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
import kagglehub
# Part 1: Data Loading and Basic Exploration


#     Download the dataset
# path = kagglehub.dataset_download("https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv")
    
    # Load the metadata.csv file
df = pd.read_csv("/home/meku/Documents/metadata_covid.csv")  #I used the first 250 rows of data form "https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv"

def basic_exploration(df):
    """
    2. Basic data exploration
    """
    # Examine the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Check DataFrame dimensions
    print(f"\nDataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Identify data types of each column
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values in important columns
    important_cols = ['title', 'abstract', 'publish_time', 'journal', 'authors']
    print("\nMissing values in important columns:")
    print(df[important_cols].isnull().sum())
    
    # Generate basic statistics for numerical columns
    print("\nBasic statistics for numerical columns:")
    print(df.describe())

# Part 2: Data Cleaning and Preparation

def handle_missing_data(df):
    """
    3. Handle missing data
    """
    # Identify columns with many missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("Percentage of missing values per column:")
    print(missing_percentage.sort_values(ascending=False))
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Remove columns with too many missing values (threshold: 80% missing)
    columns_to_drop = missing_percentage[missing_percentage > 80].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    print(f"Dropped columns: {list(columns_to_drop)}")
    
    # Fill missing values in important columns
    df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
    df_clean['title'] = df_clean['title'].fillna('Untitled')
    
    return df_clean

def prepare_data(df):
    """
    4. Prepare data for analysis
    """
    df_prepared = df.copy()
    
    # Convert date columns to datetime format
    try:
        df_prepared['publish_time'] = pd.to_datetime(df_prepared['publish_time'], errors='coerce')
    except:
        print("Could not convert publish_time to datetime")
    
    # Extract year from publication date
    df_prepared['publication_year'] = df_prepared['publish_time'].dt.year
    
    # Create abstract word count column
    df_prepared['abstract_word_count'] = df_prepared['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    # Create title word count column
    df_prepared['title_word_count'] = df_prepared['title'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    return df_prepared

# Part 3: Data Analysis and Visualization

def perform_analysis(df):
    """
    5. Perform basic analysis
    """
    analysis_results = {}
    
    # Count papers by publication year
    yearly_counts = df['publication_year'].value_counts().sort_index()
    analysis_results['yearly_counts'] = yearly_counts
    
    # Identify top journals
    top_journals = df['journal'].value_counts().head(10)
    analysis_results['top_journals'] = top_journals
    
    # Find most frequent words in titles
    all_titles = ' '.join(df['title'].dropna().astype(str))
    # Clean text: remove special characters and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', all_titles.lower())
    word_freq = Counter(words).most_common(20)
    analysis_results['title_word_freq'] = word_freq
    
    return analysis_results

def create_visualizations(df, analysis_results):
    """
    6. Create visualizations
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Plot 1: Number of publications over time
    plt.figure(figsize=(12, 6))
    analysis_results['yearly_counts'].plot(kind='line', marker='o')
    plt.title('Number of COVID-19 Publications Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('publications_over_time.png')
    plt.show()
    
    # Plot 2: Bar chart of top publishing journals
    plt.figure(figsize=(12, 6))
    analysis_results['top_journals'].plot(kind='bar')
    plt.title('Top 10 Journals Publishing COVID-19 Research')
    plt.xlabel('Journal')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_journals.png')
    plt.show()
    
    # Plot 3: Word cloud of paper titles
    all_titles = ' '.join(df['title'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Paper Titles')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('title_wordcloud.png')
    plt.show()
    
    # Plot 4: Distribution of paper counts by source
    plt.figure(figsize=(10, 6))
    df['source_x'].value_counts().head(15).plot(kind='bar')
    plt.title('Distribution of Papers by Source')
    plt.xlabel('Source')
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('source_distribution.png')
    plt.show()

# Part 4: Streamlit Application

def create_streamlit_app(df, analysis_results):
    """
    7. Build a simple Streamlit app
    """
    # Set page configuration
    st.set_page_config(
        page_title="CORD-19 Data Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # App title and description
    st.title("📚 CORD-19 Dataset Analysis Dashboard")
    st.markdown("""
    This dashboard provides insights into the COVID-19 Open Research Dataset (CORD-19) metadata.
    Explore publication trends, top journals, and word frequency analysis.
    """)
    
    # Sidebar with interactive widgets
    st.sidebar.header("🔧 Controls")
    
    # Year range slider
    if 'publication_year' in df.columns:
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(df['publication_year'].min()),
            max_value=int(df['publication_year'].max()),
            value=(2019, 2023)
        )
    
    # Journal dropdown
    top_journals_list = analysis_results['top_journals'].index.tolist()
    selected_journal = st.sidebar.selectbox(
        "Select Journal",
        options=["All"] + top_journals_list
    )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📈 Publication Trends")
        # Display publications over time plot
        st.image('publications_over_time.png')
        
        st.header("🏆 Top Journals")
        # Display top journals
        st.dataframe(analysis_results['top_journals'])
    
    with col2:
        st.header("☁️ Title Word Cloud")
        # Display word cloud
        st.image('title_wordcloud.png')
        
        st.header("📊 Source Distribution")
        # Display source distribution
        st.image('source_distribution.png')
    
    # Show sample data
    st.header("📋 Sample Data")
    st.dataframe(df.head(10))
    
    # Additional statistics
    st.header("📊 Dataset Statistics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Papers", len(df))
    
    with col4:
        st.metric("Columns", df.shape[1])
    
    with col5:
        avg_words = df['abstract_word_count'].mean() if 'abstract_word_count' in df.columns else 0
        st.metric("Avg Abstract Words", f"{avg_words:.1f}")

# Main execution
if __name__ == "__main__":
    # Part 1: Data Loading and Basic Exploration
    print("=== PART 1: DATA LOADING AND BASIC EXPLORATION ===")
    df = pd.read_csv("/home/meku/Documents/metadata_covid.csv")  
    
    if df is not None:
        basic_exploration(df)
        
        # Part 2: Data Cleaning and Preparation
        print("\n=== PART 2: DATA CLEANING AND PREPARATION ===")
        df_clean = handle_missing_data(df)
        df_prepared = prepare_data(df_clean)
        
        # Part 3: Data Analysis and Visualization
        print("\n=== PART 3: DATA ANALYSIS AND VISUALIZATION ===")
        analysis_results = perform_analysis(df_prepared)
        create_visualizations(df_prepared, analysis_results)
        
        # Part 4: Streamlit Application
        print("\n=== PART 4: STREAMLIT APPLICATION ===")
        print("To run the Streamlit app, execute: streamlit run your_script_name.py")
        print("The app code is ready in the create_streamlit_app() function.")
        
        # Uncomment the next line to automatically run Streamlit (might not work in all environments)
        # create_streamlit_app(df_prepared, analysis_results)
        
    else:
        print("Failed to load data. Please check your connection or file path.")