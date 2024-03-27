import pandas as pd
import nltk
import re
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, 
from gensim.models import Word2Vec
def extract_columns(input_csv, output_csv):
    """
    Extracts specific columns from a CSV file and saves them to a new CSV file.

    Args:
    input_csv (str): Path to the input CSV file.
    output_csv (str): Path to save the output CSV file.

    Returns:
    None
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv)

        # Extract the desired columns
        extracted_data = df[['book_id', 'original_title', 'description']]

        # Save the extracted data to a new CSV file
        extracted_data.to_csv(output_csv, index=False)
        
        print("Extraction completed successfully.")
    except Exception as e:
        print("An error occurred:", e)

def clean(desc):
    line = desc.strip()
    line = re.sub(r'[^\w\s]', '',line)
    line = re.sub("[0123456789]", '', line)
    line = re.sub(r'[\W_]+', ' ', line.lower())
    







def main():
    input_file = "books_enriched.csv" 
    output_file = "output.csv"
    extract_columns(input_file, output_file)
    three_column = pd.read_csv(output_file)






if __name__ == "__main__":
    main()

