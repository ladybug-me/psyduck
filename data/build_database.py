#!/usr/bin/env python3
"""
Psyduck - Database and Vector Index Builder
Reads data.csv and generates SQLite database (fashion.db) and FAISS index (fashion.index).
"""

import os
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data.csv')
DB_FILE = os.path.join(BASE_DIR, 'fashion.db')
INDEX_FILE = os.path.join(BASE_DIR, 'fashion.index')
EMBEDDING_MODEL = os.getenv('EMBED_MODEL', 'all-mpnet-base-v2')


def check_existing_db():
    """Check if database and FAISS index already exist and are populated."""
    if os.path.exists(DB_FILE) and os.path.exists(INDEX_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            count = cursor.execute("SELECT COUNT(*) FROM products;").fetchone()[0]
            conn.close()
            index = faiss.read_index(INDEX_FILE)
            if count > 0 and index.ntotal > 0:
                return True, count, index.ntotal
        except Exception:
            pass
    return False, 0, 0


def create_sqlite_db():
    """
    Reads data.csv and creates a clean SQLite database (fashion.db)
    with a 'products' table.
    """
    print(f"\n--- Step 1/3: Reading {DATA_FILE} ---")
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] '{DATA_FILE}' not found. Please download it first (e.g. via download_data.py).")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"[ERROR] Reading '{DATA_FILE}': {e}")
        sys.exit(1)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Use row index as unique ID, matching FAISS index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)

    # Select only required columns
    df_products = df[['id', 'image', 'display name', 'category', 'description']].copy()
    df_products['description'] = df_products['description'].fillna('')

    print(f"--- Step 2/3: Building SQLite Database ({DB_FILE}) ---")
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    df_products.to_sql('products', conn, if_exists='replace', index=False)
    conn.execute('CREATE UNIQUE INDEX idx_products_id ON products (id);')
    conn.close()

    print(f"[SUCCESS] Created '{DB_FILE}' with {len(df_products):,} products.")
    return df_products


def create_faiss_index(df_products):
    """
    Creates vector embeddings for all descriptions and saves them
    to a FAISS index file (fashion.index).
    """
    print(f"\n--- Step 3/3: Building FAISS Index ({INDEX_FILE}) ---")

    print(f"[INFO] Loading embedding model '{EMBEDDING_MODEL}'...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        sys.exit(1)

    d = model.get_sentence_embedding_dimension()
    print(f"[INFO] Model dimension: {d}")

    descriptions = df_products['description'].tolist()
    print(f"[INFO] Creating {len(descriptions):,} embeddings (may take several minutes)...")
    embeddings = model.encode(descriptions, show_progress_bar=True, convert_to_tensor=False)

    print("[INFO] Building FAISS IndexIDMap...")
    index_flat = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(index_flat)

    product_ids = np.array(df_products['id'].values, dtype='int64')
    index.add_with_ids(np.array(embeddings).astype('float32'), product_ids)

    faiss.write_index(index, INDEX_FILE)
    print(f"[SUCCESS] Created '{INDEX_FILE}'.")


def main():
    parser = argparse.ArgumentParser(description="Build Psyduck SQLite database and FAISS index.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if database and index already exist")
    args = parser.parse_args()

    exists, db_cnt, idx_cnt = check_existing_db()
    if exists and not args.force:
        print(f"[INFO] Database and FAISS index are already built:")
        print(f"  - Database ({DB_FILE}): {db_cnt:,} products")
        print(f"  - FAISS Index ({INDEX_FILE}): {idx_cnt:,} vectors")
        print("Use --force to rebuild.")
        sys.exit(0)

    print("--- Starting Database and FAISS Index Setup ---")
    products_df = create_sqlite_db()
    if products_df is not None:
        create_faiss_index(products_df)
        print("\n--- DATABASE SETUP COMPLETE ---")
        print(f"Assets generated:")
        print(f"1. {DB_FILE} (Product metadata)")
        print(f"2. {INDEX_FILE} (Semantic search index)")
    else:
        print("[ERROR] Setup failed. Could not create database.")
        sys.exit(1)


if __name__ == "__main__":
    main()