import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import os


class BiasClassifier:
    def __init__(self, model_path='./fine-tuned-multiclass-bias-model'):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        self.label_map = {
            0: "Gender Bias",
            1: "Neutral",
            2: "Political Bias",
            3: "Professional Bias",
            4: "Racial Bias",
            5: "Religious Bias"
        }
        
        print("Available bias categories:")
        for label_id, category in self.label_map.items():
            print(f" {label_id}: {category}")

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_class = self.label_map.get(predicted_class_id, "Unknown")
        
        confidence = probabilities[0][predicted_class_id].item()
        
        all_probabilities = {}
        for class_id, class_name in self.label_map.items():
            all_probabilities[class_name] = probabilities[0][class_id].item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }


class BiasCorrector:
    def __init__(self, gemini_api_key):
        self.api_key = gemini_api_key
        genai.configure(api_key=self.api_key)
        
        try:
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("Gemini 2.0 Flash API initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            self.model = None

    def get_unbiased_version(self, sentence, bias_type, confidence):
        if not self.model:
            return "Error: Gemini API not available"
        
        if bias_type == "Neutral":
            return "The sentence appears to be neutral. No changes needed."
        
        prompt = f"""
        You are an expert in identifying and correcting bias in text. I have a sentence that has been detected to contain {bias_type} with a confidence of {confidence:.3f}.

        Original sentence: "{sentence}"
        Detected bias type: {bias_type}
        
        Please provide an unbiased, neutral version of this sentence that:
        1. Maintains the original meaning and intent
        2. Removes any {bias_type.lower()} elements
        3. Uses inclusive and neutral language
        4. Preserves the factual content
        5. Is grammatically correct and natural sounding
        
        Only return the corrected sentence, nothing else.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating unbiased version: {e}"


def main():
    GEMINI_API_KEY = "AIzaSyD_5dys-RtAW1FHd-wfgkqvbhvSdjzTwmg"
    
    if not GEMINI_API_KEY:
        print("Error: Please enter your Gemini API key in the GEMINI_API_KEY variable")
        return
    
    model_path = './fine-tuned-multiclass-bias-model'
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at '{model_path}'")
        print("Please make sure you've run the training script first.")
        return
    
    print("Initializing Bias Detection and Correction System...")
    print("="*70)
    
    classifier = BiasClassifier(model_path)
    corrector = BiasCorrector(GEMINI_API_KEY)
    
    print("\n" + "="*70)
    print("BIAS DETECTION AND CORRECTION SYSTEM")
    print("Powered by BERT + Gemini 2.0 Flash")
    print("="*70)
    print("Enter sentences to detect bias and get unbiased versions.")
    print("Type 'quit' or 'exit' to stop.")
    print("-"*70)
    
    while True:
        user_input = input("\nEnter a sentence: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a valid sentence.")
            continue
        
        print("\n" + "-" * 60)
        print(f"Analyzing: \"{user_input}\"")
        print("-" * 60)
        
        print(" Step 1: Analyzing for bias using BERT model...")
        result = classifier.predict(user_input)
        
        print(f"\n BIAS ANALYSIS RESULTS:")
        print(f"   Detected Bias: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        sorted_probs = sorted(result['all_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
        print(f"\n   Top 3 Classifications:")
        for i, (class_name, prob) in enumerate(sorted_probs, 1):
            print(f"   {i}. {class_name}: {prob:.3f}")
        
        print(f"\n Step 2: Generating unbiased version using Gemini 2.0 Flash...")
        unbiased_sentence = corrector.get_unbiased_version(
            user_input, 
            result['predicted_class'], 
            result['confidence']
        )
        
        print(f"\n UNBIASED VERSION:")
        print(f"   \"{unbiased_sentence}\"")
        
        if result['predicted_class'] != "Neutral":
            print(f"\n COMPARISON:")
            print(f"   Original:  \"{user_input}\"")
            print(f"   Unbiased:  \"{unbiased_sentence}\"")
            print(f"   Bias Type: {result['predicted_class']} (Confidence: {result['confidence']:.3f})")
        
        print("\n" + "=" * 60)


def batch_process_file(input_file, output_file=None):
    GEMINI_API_KEY = ""
    
    if not GEMINI_API_KEY:
        print("Error: Please enter your Gemini API key")
        return
    
    classifier = BiasClassifier()
    corrector = BiasCorrector(GEMINI_API_KEY)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(sentences)} sentences from '{input_file}'...")
    print("Using Gemini 2.0 Flash for bias correction...")
    
    results = []
    for i, sentence in enumerate(sentences, 1):
        print(f"Processing {i}/{len(sentences)}: {sentence[:50]}...")
        
        bias_result = classifier.predict(sentence)
        
        unbiased_version = corrector.get_unbiased_version(
            sentence,
            bias_result['predicted_class'],
            bias_result['confidence']
        )
        
        result = {
            'original_sentence': sentence,
            'detected_bias': bias_result['predicted_class'],
            'confidence': bias_result['confidence'],
            'unbiased_version': unbiased_version,
            'all_probabilities': bias_result['all_probabilities']
        }
        results.append(result)
    
    if output_file:
        import pandas as pd
        
        df_data = []
        for result in results:
            row = {
                'original_sentence': result['original_sentence'],
                'detected_bias': result['detected_bias'],
                'confidence': result['confidence'],
                'unbiased_version': result['unbiased_version']
            }
            for bias_type, prob in result['all_probabilities'].items():
                row[f'{bias_type}_probability'] = prob
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False)
        print(f"Results saved to '{output_file}'")
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING RESULTS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Original: \"{result['original_sentence']}\"")
        print(f"   Bias: {result['detected_bias']} (Confidence: {result['confidence']:.3f})")
        print(f"   Unbiased: \"{result['unbiased_version']}\"")
        print("-" * 60)


def test_system():
    GEMINI_API_KEY = "AIzaSyD_5dys-RtAW1FHd-wfgkqvbhvSdjzTwmg"
    
    if not GEMINI_API_KEY:
        print("Error: Please enter your Gemini API key for testing")
        return
    
    classifier = BiasClassifier()
    corrector = BiasCorrector(GEMINI_API_KEY)
    
    test_sentences = [
        "Women are naturally better at taking care of children than men.",
        "The doctor examined his patient while the nurse prepared her medications.",
        "All politicians are corrupt and cannot be trusted.",
        "He is a great programmer for someone from that country.",
        "The company hired him because they needed diversity in their team."
    ]
    
    print("Testing Bias Detection and Correction System")
    print("Using Gemini 2.0 Flash for corrections")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}: \"{sentence}\"")
        
        result = classifier.predict(sentence)
        print(f"Detected: {result['predicted_class']} ({result['confidence']:.3f})")
        
        correction = corrector.get_unbiased_version(
            sentence, result['predicted_class'], result['confidence']
        )
        print(f"Corrected: \"{correction}\"")
        print("-" * 60)


if __name__ == "__main__":
    main()
