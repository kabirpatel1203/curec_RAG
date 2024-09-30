import spacy
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict



def setup_ner_model():
    nlp = spacy.load("en_core_sci_md")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp

def get_umls_category(cui, linker):
    tuis = linker.kb.cui_to_entity[cui].types
    for tui in tuis:
        category = UMLS_TUI_TO_CATEGORY.get(tui)
        if category:
            return category
    return "OTHER"

# Refined mapping of UMLS Semantic Types to more accurate categories
UMLS_TUI_TO_CATEGORY = {
    # Anatomy
    'T017': 'ANATOMICAL_STRUCTURE',  # Anatomical Structure
    'T029': 'ANATOMICAL_STRUCTURE',  # Body Location or Region
    'T023': 'ANATOMICAL_STRUCTURE',  # Body Part, Organ, or Organ Component
    'T030': 'ANATOMICAL_STRUCTURE',  # Body Space or Junction
    'T031': 'ANATOMICAL_STRUCTURE',  # Body Substance
    'T022': 'ANATOMICAL_STRUCTURE',  # Body System
    'T025': 'ANATOMICAL_STRUCTURE',  # Cell
    'T026': 'ANATOMICAL_STRUCTURE',  # Cell Component
    'T018': 'ANATOMICAL_STRUCTURE',  # Embryonic Structure
    'T021': 'ANATOMICAL_STRUCTURE',  # Fully Formed Anatomical Structure
    'T024': 'ANATOMICAL_STRUCTURE',  # Tissue

    # Chemicals & Drugs
    'T116': 'CHEMICAL',              # Amino Acid, Peptide, or Protein
    'T195': 'CHEMICAL',              # Antibiotic
    'T123': 'CHEMICAL',              # Biologically Active Substance
    'T122': 'CHEMICAL',              # Biomedical or Dental Material
    'T103': 'CHEMICAL',              # Chemical
    'T120': 'CHEMICAL',              # Chemical Viewed Functionally
    'T104': 'CHEMICAL',              # Chemical Viewed Structurally
    'T200': 'CHEMICAL',              # Clinical Drug
    'T196': 'CHEMICAL',              # Element, Ion, or Isotope
    'T126': 'CHEMICAL',              # Enzyme
    'T131': 'CHEMICAL',              # Hazardous or Poisonous Substance
    'T125': 'CHEMICAL',              # Hormone
    'T129': 'CHEMICAL',              # Immunologic Factor
    'T130': 'CHEMICAL',              # Indicator, Reagent, or Diagnostic Aid
    'T197': 'CHEMICAL',              # Inorganic Chemical
    'T114': 'CHEMICAL',              # Nucleic Acid, Nucleoside, or Nucleotide
    'T109': 'CHEMICAL',              # Organic Chemical
    'T121': 'CHEMICAL',              # Pharmacologic Substance
    'T192': 'CHEMICAL',              # Receptor
    'T127': 'CHEMICAL',              # Vitamin

    # Diseases and Conditions
    'T020': 'DISEASE',               # Acquired Abnormality
    'T190': 'DISEASE',               # Anatomical Abnormality
    'T049': 'DISEASE',               # Cell or Molecular Dysfunction
    'T019': 'DISEASE',               # Congenital Abnormality
    'T047': 'DISEASE',               # Disease or Syndrome
    'T050': 'DISEASE',               # Experimental Model of Disease
    'T037': 'INJURY',                # Injury or Poisoning
    'T048': 'MENTAL_DISORDER',       # Mental or Behavioral Dysfunction
    'T191': 'DISEASE',               # Neoplastic Process
    'T046': 'DISEASE',               # Pathologic Function

    # Signs and Symptoms
    'T033': 'FINDING',               # Finding
    'T184': 'SYMPTOM',               # Sign or Symptom

    # Physiology
    'T039': 'PHYSIOLOGY',            # Physiologic Function
    'T040': 'PHYSIOLOGY',            # Organism Function
    'T041': 'PHYSIOLOGY',            # Mental Process
    'T042': 'PHYSIOLOGY',            # Organ or Tissue Function
    'T043': 'PHYSIOLOGY',            # Cell Function
    'T044': 'PHYSIOLOGY',            # Molecular Function
    'T045': 'PHYSIOLOGY',            # Genetic Function

    # Procedures
    'T060': 'PROCEDURE',             # Diagnostic Procedure
    'T065': 'PROCEDURE',             # Educational Activity
    'T058': 'PROCEDURE',             # Health Care Activity
    'T059': 'PROCEDURE',             # Laboratory Procedure
    'T063': 'PROCEDURE',             # Molecular Biology Research Technique
    'T062': 'PROCEDURE',             # Research Activity
    'T061': 'PROCEDURE',             # Therapeutic or Preventive Procedure

    # Living Beings
    'T008': 'LIVING_BEING',          # Animal
    'T007': 'LIVING_BEING',          # Bacterium
    'T099': 'LIVING_BEING',          # Family Group
    'T013': 'LIVING_BEING',          # Fish
    'T004': 'LIVING_BEING',          # Fungus
    'T096': 'LIVING_BEING',          # Group
    'T016': 'LIVING_BEING',          # Human
    'T015': 'LIVING_BEING',          # Mammal
    'T001': 'LIVING_BEING',          # Organism
    'T101': 'LIVING_BEING',          # Patient or Disabled Group
    'T002': 'LIVING_BEING',          # Plant
    'T098': 'LIVING_BEING',          # Population Group
    'T097': 'LIVING_BEING',          # Professional or Occupational Group
    'T014': 'LIVING_BEING',          # Reptile
    'T011': 'LIVING_BEING',          # Amphibian
    'T005': 'LIVING_BEING',          # Virus

    # Objects
    'T071': 'OBJECT',                # Entity
    'T073': 'OBJECT',                # Manufactured Object
    'T168': 'OBJECT',                # Food
    'T072': 'OBJECT',                # Physical Object
    'T074': 'MEDICAL_DEVICE',        # Medical Device
    'T075': 'OBJECT',                # Research Device

    # Concepts & Ideas
    'T169': 'CONCEPT',               # Functional Concept
    'T102': 'CONCEPT',               # Group Attribute
    'T078': 'CONCEPT',               # Idea or Concept
    'T170': 'CONCEPT',               # Intellectual Product
    'T171': 'CONCEPT',               # Language
    'T080': 'CONCEPT',               # Qualitative Concept
    'T081': 'CONCEPT',               # Quantitative Concept
    'T089': 'CONCEPT',               # Regulation or Law
    'T082': 'CONCEPT',               # Spatial Concept
    'T079': 'CONCEPT',               # Temporal Concept

    # Activities & Behaviors
    'T052': 'ACTIVITY',              # Activity
    'T053': 'ACTIVITY',              # Behavior
    'T056': 'ACTIVITY',              # Daily or Recreational Activity
    'T051': 'ACTIVITY',              # Event
    'T064': 'ACTIVITY',              # Governmental or Regulatory Activity
    'T055': 'ACTIVITY',              # Individual Behavior
    'T066': 'ACTIVITY',              # Machine Activity
    'T057': 'ACTIVITY',              # Occupational Activity
    'T054': 'ACTIVITY',              # Social Behavior

    # Organizations
    'T093': 'ORGANIZATION',          # Health Care Related Organization
    'T092': 'ORGANIZATION',          # Organization
    'T094': 'ORGANIZATION',          # Professional Society
    'T095': 'ORGANIZATION',          # Self-help or Relief Organization

    # Geographic Areas
    'T083': 'GEOGRAPHIC_AREA',       # Geographic Area
}

def extract_medical_terms(text, nlp):
    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")
    
    entities = []
    for ent in doc.ents:
        if ent._.kb_ents:
            cui, score = ent._.kb_ents[0]
            category = get_umls_category(cui, linker)
            entities.append({
                'Term': ent.text,
                'Category': category,
                'UMLS Concept ID': cui,
                'Similarity Score': score
            })
    return entities

# # Usage
# nlp = setup_ner_model()
# text = """
# The patient was diagnosed with type 2 diabetes and was prescribed metformin.
# He also had a history of hypertension and suffered from a fracture in his left arm.
# """
# entities = extract_medical_terms(text, nlp)

# # Print the extracted entities
# for entity in entities:
#     print(f"Term: {entity['Term']}")
#     print(f"Category: {entity['Category']}")
#     print(f"UMLS Concept ID: {entity['UMLS Concept ID']}")
#     print(f"Similarity Score: {entity['Similarity Score']:.2f}")
#     print()



