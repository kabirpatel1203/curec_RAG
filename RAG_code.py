import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def setup_rag_system(reference_file_path):
    df = pd.read_csv(reference_file_path)
    
    # model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('gsarti/scibert-nli')
    reference_texts = df.apply(lambda row: f"{row['CODE']} {row['Medical term']} {row['Description']}", axis=1).tolist()
    reference_embeddings = []
    
    print("Generating embeddings...")
    for text in tqdm(reference_texts, desc="Embedding Progress"):
        embedding = model.encode([text])[0]
        reference_embeddings.append(embedding)
    
    reference_embeddings = np.array(reference_embeddings)
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(reference_embeddings.shape[1])
    index.add(reference_embeddings.astype('float32'))
    
    return df, model, index, reference_texts



def setup_tfidf(reference_texts):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reference_texts)
    
    return vectorizer, tfidf_matrix

def combined_lookup(keywords, df, model, index, vectorizer, tfidf_matrix, top_k=3, similarity_threshold=0.5, tfidf_threshold=0.5):
    # Encode the keywords for semantic search
    keyword_embeddings = model.encode(keywords)
    
    # Perform similarity search
    distances, indices = index.search(keyword_embeddings.astype('float32'), top_k)
    
    results = {}
    for i, keyword in enumerate(keywords):
        matches = []
        # Semantic matches
        for j in range(top_k):
            idx = indices[i][j]
            row = df.iloc[idx]
            similarity = 1 - distances[i][j] / 2  # Convert L2 distance to similarity score
            if similarity > similarity_threshold:  # Apply semantic similarity threshold
                matches.append({
                    'type': 'semantic',
                    'CODE': row['CODE'],
                    'Medical term': row['Medical term'],
                    'Description': row['Description'],
                    'Score': similarity
                })
        
        # Exact matches (TF-IDF)
        keyword_tfidf = vectorizer.transform([keyword])
        tfidf_scores = np.dot(tfidf_matrix, keyword_tfidf.T).toarray().flatten()
        tfidf_indices = np.argsort(-tfidf_scores)[:top_k]
        
        for idx in tfidf_indices:
            row = df.iloc[idx]
            tfidf_score = tfidf_scores[idx]
            if tfidf_score > tfidf_threshold:  # Apply TF-IDF score threshold
                matches.append({
                    'type': 'exact',
                    'CODE': row['CODE'],
                    'Medical term': row['Medical term'],
                    'Description': row['Description'],
                    'Score': tfidf_score
                })
        
        # Sort matches by score, prioritize exact matches
        matches = sorted(matches, key=lambda x: x['Score'], reverse=True)
        results[keyword] = matches[:top_k]  # Limit to top_k results
    
    return results

# # Usage example
# reference_file_path = 'Section111ValidICD10-Jan2024.csv'
# df, model, index, reference_texts = setup_rag_system(reference_file_path)

# # Setup TF-IDF system
# vectorizer, tfidf_matrix = setup_tfidf(reference_texts)

# # Perform combined lookup
# keywords = ["diabetes", "hypertension", "fracture", "yaws"]
# results = combined_lookup(keywords, df, model, index, vectorizer, tfidf_matrix)

# # Display the results
# for keyword, matches in results.items():
#     print(f"\nTop matches for '{keyword}':")
#     for match in matches:
#         print(f"Type: {match['type'].capitalize()}")
#         print(f"CODE: {match['CODE']}")
#         print(f"Medical term: {match['Medical term']}")
#         print(f"Score: {match['Score']:.2f}")
#         print()

