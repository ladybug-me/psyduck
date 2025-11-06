import pandas as pd
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import sys

# --- CONFIGURATION ---
DATA_FILE = 'data.csv'
DB_FILE = 'fashion.db'
INDEX_FILE = 'fashion.index'
EMBEDDING_MODEL = 'all-mpnet-base-v2' # A popular, fast, and good-quality model
# ---------------------

def create_sqlite_db():
    """
    Reads data.csv and creates a clean SQLite database (fashion.db)
    with a 'products' table.
    """
    print(f"\n--- Step 1/3: Reading {DATA_FILE} ---")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: '{DATA_FILE}' not found. Please download it first.")
        sys.exit()

    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Use the row index as the unique ID, as it's simple and
    # will match the FAISS index perfectly.
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    
    # Select only the columns we need for the app
    # We keep 'description' for the Ollama re-ranking step
    df_products = df[['id', 'image', 'display name', 'category', 'description']]
    
    # Handle missing descriptions, which can cause errors
    df_products['description'] = df_products['description'].fillna('')

    print(f"--- Step 2/3: Building SQLite Database ({DB_FILE}) ---")
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    conn = sqlite3.connect(DB_FILE)
    
    # Save the DataFrame to the SQL table
    df_products.to_sql('products', conn, if_exists='replace', index=False)
    
    # Create an index on the 'id' column for fast lookups
    conn.execute('CREATE UNIQUE INDEX idx_products_id ON products (id);')
    conn.close()
    
    print(f"Successfully created '{DB_FILE}' with {len(df_products)} products.")
    return df_products

def create_faiss_index(df_products):
    """
    Creates vector embeddings for all descriptions and saves them
    to a FAISS index file (fashion.index).
    """
    print(f"\n--- Step 3/3: Building FAISS Index ({INDEX_FILE}) ---")
    
    # 1. Load the embedding model
    print(f"Loading embedding model '{EMBEDDING_MODEL}'... (This may download the model)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Get the embedding dimension (e.g., all-MiniLM-L6-v2 is 384)
    d = model.get_sentence_embedding_dimension()
    print(f"Model dimension: {d}")

    # 2. Get descriptions to embed
    descriptions = df_products['description'].tolist()
    
    # 3. Create embeddings
    print(f"Creating {len(descriptions)} embeddings... (This is the slow part, may take several minutes)")
    embeddings = model.encode(descriptions, show_progress_bar=True, convert_to_tensor=False)

    # 4. Build the FAISS Index
    print("Building FAISS index...")
    # We use IndexIDMap to map our custom IDs (0, 1, 2, ...) to the vectors
    # This lets FAISS return our SQLite product IDs directly.
    index_flat = faiss.IndexFlatL2(d)  # The core search algorithm
    index = faiss.IndexIDMap(index_flat) # The ID wrapper
    
    # Get the product IDs to map to the vectors
    product_ids = np.array(df_products['id'].values, dtype='int64')

    # 5. Add vectors and their corresponding IDs to the index
    index.add_with_ids(np.array(embeddings).astype('float32'), product_ids)

    # 6. Save the index to disk
    faiss.write_index(index, INDEX_FILE)
    print(f"Successfully created '{INDEX_FILE}'.")


if __name__ == "__main__":
    print("--- Starting Phase 1 Database and Index Setup ---")
    
    # Step 1 & 2: Create SQLite DB and get the dataframe
    products_df = create_sqlite_db()
    
    # Step 3: Create FAISS index from the dataframe
    if products_df is not None:
        create_faiss_index(products_df)
        print("\n--- PHASE 1 SETUP COMPLETE ---")
        print(f"Your 'behind the scenes' assets are ready:")
        print(f"1. {DB_FILE} (Your product metadata)")
        print(f"2. {INDEX_FILE} (Your semantic search index)")
    else:
        print("Setup failed. Could not create database.")