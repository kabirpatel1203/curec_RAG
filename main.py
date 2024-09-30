from NER_code import setup_ner_model, extract_medical_terms
from RAG_code import setup_rag_system, combined_lookup, setup_tfidf

class MedicalTermProcessor:
    def __init__(self):
        # Initialize NER model
        self.nlp = setup_ner_model()
        
        # Initialize RAG system
        self.df, self.model, self.index, self.reference_texts = setup_rag_system('Section111ValidICD10-Jan2024.csv')
        self.vectorizer, self.tfidf_matrix = setup_tfidf(self.reference_texts)

    def process_text(self, text):
        # Extract medical terms using NER
        entities = extract_medical_terms(text, self.nlp)
        
        # Look up codes using RAG
        keywords = [entity['Term'] for entity in entities]
        rag_results = combined_lookup(keywords, self.df, self.model, self.index, self.vectorizer, self.tfidf_matrix)
        
        # Combine NER and RAG results
        combined_results = []
        for entity in entities:
            term = entity['Term']
            combined_results.append({
                'Term': term,
                'UMLS Concept ID': entity['UMLS Concept ID'],
                'Category': entity['Category'],
                'Matches': rag_results.get(term, [])
            })
        
        return combined_results































# import spacy
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# import torch
# import pandas as pd
# from fuzzywuzzy import process

# class MedicalTermProcessor:
#     def __init__(self):
#         # Load SciBERT model
#         self.nlp = spacy.load("en_core_sci_scibert")
        
#         # Load the pre-trained ClinicalNER model and tokenizer
#         self.ner_model = AutoModelForTokenClassification.from_pretrained("Posos/ClinicalNER")
#         self.tokenizer = AutoTokenizer.from_pretrained("Posos/ClinicalNER")
        
#         # Load the CSV data
#         self.df = pd.read_csv('Section111ValidICD10-Jan2024.csv')

#     def extract_entities(self, text):
#         doc = self.nlp(text)
#         return [ent.text for ent in doc.ents]

#     def categorize_term(self, term):
#         inputs = self.tokenizer(term, return_tensors="pt")
#         outputs = self.ner_model(**inputs)
#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=2)
#         predicted_labels = [self.ner_model.config.id2label[p.item()] for p in predictions[0]]
#         return predicted_labels[0]

#     def lookup_medical_terms(self, queries, num_results=3):
#         all_results = {}
#         for query in queries:
#             matches = process.extract(query, self.df['Medical term'], limit=num_results)
#             results = []
#             for match in matches:
#                 term = match[0]
#                 row = self.df[self.df['Medical term'] == term].iloc[0]
#                 results.append({
#                     'CODE': row['CODE'],
#                     'Medical term': row['Medical term'],
#                     'Description': row['Description'],
#                     'Category': self.categorize_term(term)
#                 })
#             all_results[query] = results
#         return all_results

#     def process_text(self, text):
#         entities = self.extract_entities(text)
#         return self.lookup_medical_terms(entities)





