# create_multiclass_dataset.py
import os
import json
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_multiclass_dataset():
    """
    Creates a multi-class labeled dataset by using the BOLD filenames
    to assign specific bias categories to each sentence.
    """
    data_dir = 'data'
    output_csv = 'data/final_multiclass_data.csv'
    
    # Mapping from filename keywords to bias categories
    domain_map = {
        'religious_ideology': 'Religious Bias',
        'gender': 'Gender Bias',
        'race': 'Racial Bias',
        'political_ideology': 'Political Bias',
        'profession': 'Professional Bias'
    }

    wiki_files = [f for f in os.listdir(data_dir) if f.endswith('_wiki.json')]

    if not wiki_files:
        print("Error: No '_wiki.json' files found in 'data'.")
        return

    print(f"Processing {len(wiki_files)} BOLD wiki files...")
    
    labeled_rows = []
    
    # Process each file and assign the correct bias type
    for filename in wiki_files:
        # Find the matching domain from our map
        domain_key = next((key for key in domain_map if key in filename), None)
        if not domain_key:
            continue
            
        bias_type = domain_map[domain_key]
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)
            
        for domain in wiki_data.values():
            for sentences in domain.values():
                for sentence in sentences:
                    labeled_rows.append({'text': sentence.strip(), 'bias_type': bias_type})

    # Add neutral examples
    neutral_sentences = [
        "The scheduled maintenance will begin tomorrow morning.",
        "She decided to learn a new language.",
        "The park is a popular spot for picnics and outdoor activities.",
        "He submitted his final report before the deadline.",
        "The documentary provided a lot of interesting information."
    ]
    for sentence in neutral_sentences:
        labeled_rows.append({'text': sentence, 'bias_type': 'Neutral'})

    # --- Convert text labels to numeric labels ---
    df = pd.DataFrame(labeled_rows)
    
    # Use scikit-learn's LabelEncoder to create numeric IDs for each class
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['bias_type'])
    
    # Save the final dataset
    df[['text', 'label']].to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully created multi-class dataset at '{output_csv}'.")
    print("The model will be trained on the following classes:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  Label {i}: {class_name}")

if __name__ == "__main__":
    create_multiclass_dataset()
