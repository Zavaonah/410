import pandas as pd
import nltk
import re
import numpy as np
import string
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
nltk.download('wordnet')

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
    try:
        line = desc.strip()
        line = re.sub(r'[^\w\s]', ' ',line.lower())
        line = re.sub("[0123456789]", '', line)
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        line_tokenized = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(line)]
        return line_tokenized
    except:
        print(desc)
        print(type(desc))
    

def train_word2vec_model(cleaned):
    model = Word2Vec(cleaned, vector_size=100, window=5, min_count=1, hs=1, negative=0)
    return model


def compute_word_vectors(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        vec = np.mean(vectors, axis=0)
        return (vec / np.linalg.norm(vec))
    else:
        return np.zeros(model.vector_size)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_similarity(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    return dot_product

def compute_word_relevances_vec(query, cleaned, model):
    query_tokens = clean(query)
    query_vector = compute_word_vectors(query_tokens, model)
    
    relevances = []
    for index, row in cleaned.iterrows():
        description = row['description']
        doc_vector = compute_word_vectors(description, model)
        similarity = compute_similarity(query_vector, doc_vector)
        if row['original_title'] == "To Kill a Mockingbird":
            print("TKAM: " + str(similarity))
            print("Query: ", query_tokens)
            print("--------------------------------")
            print("Doc: ", description)
        relevances.append((row['original_title'], similarity))
    return relevances



def main():
    input_file = "books_enriched.csv" 
    output_file = "output.csv"
    extract_columns(input_file, output_file)
    three_column = pd.read_csv(output_file)
    three_column = three_column.dropna(subset=['description'])
    three_column['description'] = three_column['description'].apply(clean)
    model = train_word2vec_model(three_column['description'].to_list())
    
    queries = ["The unforgettable novel of a childhood in a sleepy Southern town and the crisis of conscience that rocked it, To Kill A Mockingbird became both an instant bestseller and a critical success when it was first published in 1960. It went on to win the Pulitzer Prize in 1961 and was later made into an Academy Award-winning film, also a classic.Compassionate, dramatic, and deeply moving, To Kill A Mockingbird takes readers to the roots of human behavior - to innocence and experience, kindness and cruelty, love and hatred, humor and pathos. Now with over 18 million copies in print and translated into forty languages, this regional story by a young Alabama woman claims universal appeal. Harper Lee always considered her book to be a simple love story. Today it is regarded as a masterpiece of American literature."]
    for query in queries:
        relevances = compute_word_relevances_vec(query, three_column, model)
        sorted_relevances = sorted(relevances, key=lambda x: x[1], reverse=True)
        print(f"\n5 Most Relevant Document Indices And Score For {query}:\n")
        for idx, relevance in sorted_relevances[:20]:
            print(f"{idx}")


    print(three_column.head())
    






if __name__ == "__main__":
    main()

