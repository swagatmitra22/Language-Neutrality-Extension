import os
import pandas as pd
import numpy as np
import torch
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasClassificationConfig:
    """Configuration for bias classification model"""
    DATASET_PATH = r'data\final_multiclass_data.csv'
    MODEL_NAME = 'distilbert-base-uncased'
    OUTPUT_DIR = './fine-tuned-multiclass-bias-model-v2'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    TRAIN_TEST_SPLIT = 0.8
    VAL_TEST_SPLIT = 0.5
    SEED = 42

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def prepare_dataset(dataset_path, config):
    """Load and split dataset into train, validation, and test sets"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    # Identify label column (typically the last column)
    label_column = df.columns[-1]
    text_column = df.columns[0] if len(df.columns) > 1 else df.columns[0]
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Text column: {text_column}, Label column: {label_column}")
    logger.info(f"Class distribution:\n{df[label_column].value_counts()}")
    
    # Create label mapping with Python int types (not numpy int64)
    unique_labels = sorted(df[label_column].unique())
    label2id = {int(label): int(idx) for idx, label in enumerate(unique_labels)}
    id2label = {int(idx): int(label) for label, idx in label2id.items()}
    
    logger.info(f"Label to ID mapping: {label2id}")
    logger.info(f"ID to Label mapping: {id2label}")
    
    # Convert labels to numeric
    df['labels'] = df[label_column].map(lambda x: int(label2id[int(x)]))
    df = df.rename(columns={text_column: 'text'})
    df = df[['text', 'labels']].reset_index(drop=True)
    
    # Ensure labels are Python ints
    df['labels'] = df['labels'].astype(int)
    
    # First split: train (80%) and temp (20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - config.TRAIN_TEST_SPLIT,
        random_state=config.SEED,
        stratify=df['labels']
    )
    
    # Second split: validation (50% of temp = 10%) and test (50% of temp = 10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.VAL_TEST_SPLIT,
        random_state=config.SEED,
        stratify=temp_df['labels']
    )
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Convert to HuggingFace Dataset objects
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[['text', 'labels']], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']], preserve_index=False)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict, label2id, id2label

def tokenize_function(examples, tokenizer, config):
    """Tokenize text examples"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=config.MAX_LENGTH
    )

def train_multiclass_model():
    """Train the multiclass bias classification model"""
    config = BiasClassificationConfig()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Set optimal CUDA settings
        torch.cuda.empty_cache()
    
    # Check dataset exists
    if not os.path.exists(config.DATASET_PATH):
        logger.error(f"Dataset not found at {config.DATASET_PATH}")
        return
    
    # Prepare dataset
    dataset_dict, label2id, id2label = prepare_dataset(config.DATASET_PATH, config)
    num_labels = len(label2id)
    logger.info(f"Number of classes: {num_labels}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        clean_up_tokenization_spaces=True
    )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_function(examples, tokenizer, config),
        batched=True,
        remove_columns=['text']
    )
    
    # Load model - use float32 to avoid FP16 issues
    logger.info(f"Loading model: {config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float32  # Use float32 instead of float16
    ).to(device)
    
    # Training arguments - FIXED: Disabled FP16 and gradient accumulation issues
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir='./logs-multiclass',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to='none',
        seed=config.SEED,
        fp16=False,  # DISABLED - Causes issues with gradient unscaling
        bf16=False,  # Disabled bfloat16
        gradient_accumulation_steps=1,  # REDUCED - No accumulation to avoid FP16 issues
        optim='adamw_torch',
        logging_first_step=True,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        dataloader_pin_memory=True if device == 'cuda' else False,
        use_cpu=False,
        max_grad_norm=1.0,
    )
    
    # Initialize trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0
            )
        ]
    )
    
    # Train model
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise
    
    # Evaluate on test set
    logger.info("="*70)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*70)
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    logger.info(f"Test Results: {test_results}")
    
    # Detailed evaluation with predictions
    test_predictions = trainer.predict(tokenized_datasets['test'])
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    test_labels = np.array(tokenized_datasets['test']['labels'])
    
    logger.info("\n" + "="*70)
    logger.info("DETAILED CLASSIFICATION REPORT")
    logger.info("="*70)
    class_names = [str(id2label[i]) for i in sorted(id2label.keys())]
    logger.info(classification_report(
        test_labels,
        test_preds,
        target_names=class_names
    ))
    
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save model and tokenizer
    logger.info(f"Saving model to {config.OUTPUT_DIR}")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    # Save config with proper Python types
    config_dict = {
        'label2id': label2id,
        'id2label': {str(k): int(v) for k, v in id2label.items()},
        'model_name': config.MODEL_NAME,
        'num_labels': num_labels,
    }
    
    config_path = os.path.join(config.OUTPUT_DIR, 'label_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to {config.OUTPUT_DIR}")
    logger.info("="*70)
    
    return trainer, model, tokenizer, id2label

def inference(text, model_path, device='cuda'):
    """Run inference on a single text"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        
        # Load label mapping
        config_path = os.path.join(model_path, 'label_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        id2label = {int(k): v for k, v in config_dict['id2label'].items()}
        
        # Tokenize
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
        
        return {
            'text': text,
            'predicted_bias_class': id2label[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {
                id2label[i]: float(torch.softmax(logits, dim=1)[0][i].item())
                for i in range(len(id2label))
            }
        }
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return None

if __name__ == "__main__":
    # Train model
    trainer, model, tokenizer, id2label = train_multiclass_model()
    
    # Example inference (uncomment to test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("RUNNING INFERENCE EXAMPLES")
    print("="*70)
    
    test_sentences = [
        "This is a test sentence",
        "Another example text for classification",
        "Sample bias detection text",
    ]
    
    for sentence in test_sentences:
        result = inference(sentence, './fine-tuned-multiclass-bias-model-v2', device=device)
        if result:
            print(f"\nText: {result['text']}")
            print(f"Predicted Class: {result['predicted_bias_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
