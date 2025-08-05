# create_final_dataset.py
import os
import json
import csv

def create_final_dataset_from_wiki():
    """
    Creates a labeled dataset using the full sentences from _wiki.json files
    for the biased class (label 1) and adds generic neutral examples (label 0).
    """
    data_dir = 'data'
    output_csv = 'data/final_labeled_data.csv'
    
    # Find all the _wiki.json files in the data directory
    wiki_files = [f for f in os.listdir(data_dir) if f.endswith('_wiki.json')]

    if not wiki_files:
        print("Error: No '_wiki.json' files found in the 'data' directory.")
        print("Please add the BOLD _wiki.json files to proceed.")
        return

    print(f"Found {len(wiki_files)} wiki files to process: {wiki_files}")

    labeled_data = []

    # Process each wiki file to extract full, biased sentences
    for filename in wiki_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)
        
        # The structure is {domain: {sub_group: [sentences]}}
        for domain in wiki_data.values():
            for sentences in domain.values():
                for sentence in sentences:
                    # Add the full sentence with a "biased" label
                    labeled_data.append([sentence.strip(), 1])
    
    print(f"Extracted {len(labeled_data)} biased examples from wiki files.")

    # Add generic neutral sentences (Label 0)
    neutral_sentences = [
        "The event was scheduled for next Tuesday.",
        "He took a photo of the sunset.",
        "The committee will review the proposal.",
        "They decided to repaint the living room.",
        "The recipe requires three eggs and a cup of flour.",
        "The cat slept peacefully on the windowsill.",
        "She found a new book to read at the library.",
        "The store is having a sale this weekend.",
        "He enjoys hiking on the weekends.",
        "The meeting was productive and ended on time."
    ]
    
    for sentence in neutral_sentences:
        labeled_data.append([sentence, 0])

    # Write all data to the final CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label']) # Header row
        writer.writerows(labeled_data)

    print(f"Successfully created final dataset at '{output_csv}' with {len(labeled_data)} total examples.")

if __name__ == "__main__":
    create_final_dataset_from_wiki()

