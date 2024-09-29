import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import pandas as pd
from fuzzywuzzy import process

class MedicalTermProcessor:
    def __init__(self):
        # Load SciBERT model
        self.nlp = spacy.load("en_core_sci_scibert")
        
        # Load the pre-trained ClinicalNER model and tokenizer
        self.ner_model = AutoModelForTokenClassification.from_pretrained("Posos/ClinicalNER")
        self.tokenizer = AutoTokenizer.from_pretrained("Posos/ClinicalNER")
        
        # Load the CSV data
        self.df = pd.read_csv('Section111ValidICD10-Jan2024.csv')

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

    def categorize_term(self, term):
        inputs = self.tokenizer(term, return_tensors="pt")
        outputs = self.ner_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        predicted_labels = [self.ner_model.config.id2label[p.item()] for p in predictions[0]]
        return predicted_labels[0]

    def lookup_medical_terms(self, queries, num_results=3):
        all_results = {}
        for query in queries:
            matches = process.extract(query, self.df['Medical term'], limit=num_results)
            results = []
            for match in matches:
                term = match[0]
                row = self.df[self.df['Medical term'] == term].iloc[0]
                results.append({
                    'CODE': row['CODE'],
                    'Medical term': row['Medical term'],
                    'Description': row['Description'],
                    'Category': self.categorize_term(term)
                })
            all_results[query] = results
        return all_results

    def process_text(self, text):
        entities = self.extract_entities(text)
        return self.lookup_medical_terms(entities)

# Usage
# processor = MedicalTermProcessor()

# def process_medical_text(text):
#     results = processor.process_text(text)
#     for query, matches in results.items():
#         print(f"Matches for '{query}':")
#         for match in matches:
#             print(f"CODE: {match['CODE']}")
#             print(f"Medical Category: {match['Category']}")
#             print(f"Medical term: {match['Medical term']}")
#             print(f"Description: {match['Description']}")
#         print()

# # Example usage
# text = """
# The patient was diagnosed with type 2 diabetes and hypertension. He is currently taking Metformin 500mg 
# twice daily and recently underwent a CT scan for abdominal pain. Symptoms include elevated blood pressure.
# """

# process_medical_text(text)