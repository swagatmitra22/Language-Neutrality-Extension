# train_multiclass_model.py
import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch

def train_multiclass_model():
    """
    Trains a BERT model for multi-class bias classification.
    """
    # --- 1. Setup and Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using training device: {device}")
    
    dataset_path = 'data/final_multiclass_data.csv'
    base_model_name = 'bert-base-uncased'
    output_model_dir = './fine-tuned-multiclass-bias-model'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at '{dataset_path}'. Please run the creation script first.")
        return

    # --- 2. Determine Number of Labels ---
    # We read the number of unique labels directly from our new dataset
    num_labels = pd.read_csv(dataset_path)['label'].nunique()
    print(f"Found {num_labels} unique labels in the dataset.")

    # --- 3. Load Dataset, Tokenizer, and Model ---
    dataset = load_dataset('csv', data_files=dataset_path)['train']
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels  # Crucial change: Tell the model how many classes to predict
    ).to(device)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # --- 4. Define Training Arguments and Trainer ---
    training_args = TrainingArguments(
        output_dir='./results-multiclass',
        num_train_epochs=4, # Increased epochs slightly for the more complex task
        per_device_train_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs-multiclass',
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    
    # --- 5. Start Training ---
    print("\nStarting multi-class model training...")
    trainer.train()
    print("\nTraining complete!")

    # --- 6. Save the Final Model ---
    print(f"Saving the multi-class model to '{output_model_dir}'...")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Multi-class model saved. Phase 2 is now complete.")

if __name__ == "__main__":
    train_multiclass_model()
