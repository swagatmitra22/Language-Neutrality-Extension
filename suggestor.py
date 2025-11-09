import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class BiasClassifier:
    """
    Detects bias type in a given sentence using a fine-tuned BERT model.
    """

    def __init__(self, model_path='./fine-tuned-multiclass-bias-model'):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model from {model_path}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

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
            print(f"  {label_id}: {category}")

    def predict(self, text: str):
        """
        Predicts the type of bias in a sentence and returns confidence scores.
        """
        if not text.strip():
            return {"predicted_class": "Unknown", "confidence": 0.0, "all_probabilities": {}}

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

        all_probabilities = {
            self.label_map[i]: probabilities[0][i].item()
            for i in self.label_map
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }


class BiasCorrector:
    """
    Uses Gemini API to generate unbiased versions of biased sentences.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Missing GEMINI_API_KEY in environment variables. Add it to your .env file.")

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini 2.0 Flash API initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing Gemini API: {e}")
            self.model = None

    def get_unbiased_version(self, sentence, bias_type, confidence):
        """
        Generates an unbiased rewrite of the sentence using Gemini.
        """
        if not self.model:
            return "Error: Gemini API not available."

        if bias_type == "Neutral":
            return "The sentence appears to be neutral. No changes needed."

        prompt = f"""
        You are an expert in identifying and correcting bias in text.
        I have a sentence that has been detected to contain {bias_type} with a confidence of {confidence:.3f}.

        Original sentence: "{sentence}"
        Detected bias type: {bias_type}

        Please provide an unbiased, neutral version of this sentence that:
        1. Maintains the original meaning and intent.
        2. Removes any {bias_type.lower()} elements.
        3. Uses inclusive and neutral language.
        4. Preserves the factual content.
        5. Is grammatically correct and natural sounding.

        Only return the corrected sentence, nothing else.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response and response.text else "(No response from Gemini.)"
        except Exception as e:
            return f"Error generating unbiased version: {e}"


def main():
    """
    CLI entry point for bias detection and correction.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ Error: Missing GEMINI_API_KEY. Add it to your .env file.")
        return

    model_path = './fine-tuned-multiclass-bias-model'
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model not found at '{model_path}'. Run the training script first.")
        return

    print("Initializing Bias Detection and Correction System...")
    print("=" * 70)

    classifier = BiasClassifier(model_path)
    corrector = BiasCorrector()

    print("\n" + "=" * 70)
    print("BIAS DETECTION AND CORRECTION SYSTEM")
    print("Powered by BERT + Gemini 2.0 Flash")
    print("=" * 70)
    print("Enter sentences to detect bias and get unbiased versions.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 70)

    while True:
        user_input = input("\nEnter a sentence: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break

        if not user_input:
            print("⚠️ Please enter a valid sentence.")
            continue

        print("\n" + "-" * 60)
        print(f"Analyzing: \"{user_input}\"")
        print("-" * 60)

        # Step 1: Bias Detection
        result = classifier.predict(user_input)
        print(f"\n BIAS ANALYSIS RESULTS:")
        print(f"   Detected Bias: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.3f}")

        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n   Top 3 Classifications:")
        for i, (class_name, prob) in enumerate(sorted_probs, 1):
            print(f"   {i}. {class_name}: {prob:.3f}")

        # Step 2: Generate unbiased version
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


if __name__ == "__main__":
    main()
