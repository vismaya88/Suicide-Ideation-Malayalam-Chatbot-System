import pandas as pd
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import gradio as gr
import os
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

severity_model = None
severity_tokenizer = None
severity_label_encoder = None

tokenizer = None
model = None



# Dictionary of default responses based on keyword severity
default_responses = {
    "anxiety": "താങ്കളുടെ ആശങ്ക മനസ്സിലാകുന്നു. ദയവായി ആഴത്തിൽ ശ്വസിച്ച് ശരീരം ശാന്തമാക്കാൻ ശ്രമിക്കൂ.",
    "depression": "മനസ്സിന്റെ ക്ഷീണം അതിരുചെയ്യുന്നു. ഒരാളുമായി സംസാരിക്കാൻ ശ്രമിക്കൂ, ഒറ്റയാണെന്ന് കരുതേണ്ട.",
    "suicidal": "ഈ സംഭാഷണം അത്യന്തം ഗൗരവമേറിയതാണ്. ദയവായി ഉടൻ സഹായം തേടുക. താങ്കൾ തനിച്ചല്ല."
}

severity_keywords = {
    "anxiety": ["ഉറക്കം", "ചിന്ത", "അശാന്തത", "ഉളുപ്പ്", "തളർച്ച", "നട്ടു", "ഇളകുന്നു"],
    "depression": ["വേദന", "ഇച്ഛയില്ലായ്മ", "നിരാശ", "ദു:ഖം", "തളർന്നിരിക്കുന്നു", "മനോവൈകല്യം"],
    "suicidal": ["ചാകണം", "ജീവിതം വേണ്ട", "സമാപനം", "ഉപേക്ഷിക്കാൻ", "അവസാനം", "ജീവന്"]
}
intent_classifier = None
vectorizer = None
sentiment_lexicon = {"ഉല്ലാസം": 1, "പ്രിയം": 1, "വേദന": -1, "ദു:ഖം": -1, "നിസ്സാരത": -1}

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
memory_bank = {
    "തളർച്ച": "ജീവിതം വെറുപ്പുണ്ട് എന്ന അനുഭവം മനസ്സിലാക്കുന്നു. സഹായം തേടുക.",
    "അശാന്തത": "ശാന്തമാകാൻ ദീർഘശ്വാസം എടുക്കുക. നിങ്ങൾക്ക് ഇതിൽ നിന്ന് മാറാൻ കഴിയും.",
    "ചാകണം": "താങ്കളുടെ ജീവൻ അത്യന്തം വിലപ്പെട്ടതാണ്. ഉടൻ സഹായം തേടുക."
}
df = pd.read_excel(r"C:\Users\Devika\Desktop\Study\Sem6\NLP\Project\Suicide_detection_chatbot\therapy_malayalam_cleaned_utf8.xlsx")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['inputs'].astype(str))

def load_dataset(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(columns={"inputs": "input_text", "targets": "target_text"})
    return Dataset.from_pandas(df[["input_text", "target_text"]])

def tokenize_function(example):
    model_inputs = tokenizer(example["input_text"], max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(example["target_text"], max_length=128, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(file_path):
    print("🔄 Loading data...")
    dataset = load_dataset(file_path)
    tokenized_dataset = dataset.map(tokenize_function)
    print("🧠 Starting training...")
    training_args = TrainingArguments(
        output_dir="./mt5_malayalam_chatbot",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model)
    )
    trainer.train()

    trainer.save_model(r"C:\Users\Devika\Desktop\Study\Sem6\NLP\Project\Suicide_detection_chatbot\mt5_malayalam_chatbot")
    tokenizer.save_pretrained(r"C:\Users\Devika\Desktop\Study\Sem6\NLP\Project\Suicide_detection_chatbot\mt5_malayalam_chatbot")
    print("✅ Training complete and model saved.")


def get_similar_response(text):
    embeddings = embedding_model.encode([text] + list(memory_bank.keys()), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    best_idx = torch.argmax(cosine_scores).item()
    if cosine_scores[0][best_idx] > 0.7:
        return list(memory_bank.values())[best_idx]
    return None

def load_model():
    global tokenizer, model,  vectorizer
    if os.path.exists(r"C:\Users\Devika\Desktop\Study\Sem6\NLP\Project\Suicide_detection_chatbot\mt5_malayalam_chatbot"):
        model = MT5ForConditionalGeneration.from_pretrained(r"C:\Users\Devika\Desktop\Study\Sem6\NLP\Project\Suicide_detection_chatbot\mt5_malayalam_chatbot")
        tokenizer = MT5Tokenizer.from_pretrained(r"C:\Users\Devika\Desktop\Study\Sem6\NLP\Project\Suicide_detection_chatbot\mt5_malayalam_chatbot")
    else:
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
   
    
        




# Sample phrases representing each severity class
severity_samples = {
    "DEPRESSION": [
        "എനിക്ക് ഒന്നും മനസ്സിലാകുന്നില്ല",
        "സഹായം വേണം",
        "ദു:ഖം ഉണ്ട്",
        "ജീവിതം ബുദ്ധിമുട്ടിലാണ്",
        "ഞാന്‍ തളര്‍ന്നിരിക്കുന്നു"
    ],
    "ANXIETY": [
        "എനിക്ക് പേടിയാണ്",
        "ഞാന്‍ അലോസരത്തിലാണ്",
        "അവസാനം എന്താവും എന്ന് ഭയമാണ്",
        "മനസ്സ് ശാന്തമല്ല",
        "വികാരപരമായ ഉളഞ്ഞിരിക്കുന്നു"
    ],
    "SUICIDAL": [
        "ഞാന്‍ ജീവനെടുത്തേക്കും",
        "ജീവിതം അവസാനിപ്പിക്കണം",
        "ഞാന്‍ മരിക്കണം",
        "എന്തിന് എനിക്ക് ജീവിക്കണം",
        "ഇത് എന്നെ കൊല്ലുന്നു"
    ],
    "GENERAL": [
        "സാധാരണമായ ദിവസമാണ്",
        "എനിക്ക് ചെറിയ വിഷമം ഉണ്ട്",
        "ഒന്നും വലിയ പ്രശ്നമില്ല"
    ]
}

# Function to compute severity based on meaning
def predict_severity_semantic(text):
    input_embedding = embedding_model.encode(text, convert_to_tensor=True)

    max_score = -1
    predicted_severity = "GENERAL"

    for severity, phrases in severity_samples.items():
        for phrase in phrases:
            phrase_embedding = embedding_model.encode(phrase, convert_to_tensor=True)
            similarity = util.cos_sim(input_embedding, phrase_embedding).item()

            if similarity > max_score:
                max_score = similarity
                predicted_severity = severity

    return predicted_severity





def detect_severity(text):
    suicidal_keywords = ['ആത്മഹത്യ', 'ജീവിതം അവസാനിപ്പിക്കണം', 'ചത്തുകളയണം', 'ജീവിക്കാനാകില്ല']
    depression_keywords = ['ഉളളിലിരിക്കുക', 'ദുഃഖം', 'ചിന്ത', 'ഉളളാശയ','സഹായം വേണം']
    anxiety_keywords = ['അശാന്തി', 'ഭയം', 'പാകം', 'ഉത്കണ്ഠ', 'പേടി']
    if any(kw in text for kw in suicidal_keywords):
        return "SUICIDAL"
    elif any(kw in text for kw in depression_keywords):
        return "DEPRESSION"
    elif any(kw in text for kw in anxiety_keywords):
        return "ANXIETY"
    else:
        return "GENERAL"
    
def generate_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vec, X)
    best_match_idx = similarity_scores.argmax()
    response = df.iloc[best_match_idx]['targets']

    
    severity = predict_severity_semantic(user_input)
    rule_severity = detect_severity(user_input)
    


    print(f"Severity: {rule_severity}")
    print(f"Response: {response}")

    return f"Severity (Rule): {rule_severity}\nSeverity (Model): {severity}\n\nResponse: {response}"









def run_gradio():
    iface = gr.Interface(
        fn=generate_response,
        inputs=gr.Textbox(label=" Malayalam സന്ദേശം നൽകുക"),
        outputs=gr.Textbox(label="ചാറ്റ്ബോട്ട് പ്രതികരണവും ഗുരുതരം വിലയിരുത്തലും"),
        title="Malayalam Therapy Chatbot",
        description="സൗമ്യമായ പ്രതികരണം നൽകുന്ന മലയാളം സൈക്കോളജിക്കൽ സഹായി"
        
    )
    
    iface.launch()

if __name__ == "__main__":
    load_model()
    # Uncomment below to train once
    # train_model("therapy_malayalam_cleaned_utf8.xlsx")
    
    

    
    run_gradio()
