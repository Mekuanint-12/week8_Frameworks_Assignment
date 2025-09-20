
1. Data Loading and Basic Exploration 

Download only the metadata.csv file from the CORD-19 dataset  using the link "https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv"

I used  only the first 250 rows of data 
The python program worked to:
   . Check the DataFrame dimensions (rows, columns)

   . Identify data types of each column

   . Check for missing values in important columns

   . Generate basic statistics for numerical columns

Part 2: Data Cleaning and Preparation

the python code try to:
      Handle missing data

      Identify columns with many missing values

      Decide how to handle missing values (removal or filling)

      Create a cleaned version of the dataset

      Prepare data for analysis

      Convert date columns to datetime format

      Extract year from publication date for time-based analysis

      Create new columns if needed (e.g., abstract word count)

Part 3: Data Analysis and Visualization

The code Perform basic analysis on Count papers by publication year, Identify top journals publishing COVID-19 research.

It Create visualizations by line plot(number of publications over time), bar chart (top publishing journals), by distribution of paper counts by source and generate a word cloud of paper titles.

Part 4: Streamlit Application 

Build a simple Streamlit app

