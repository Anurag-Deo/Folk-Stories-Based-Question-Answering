"""
Folk Stories Question Answering Model Fine-tuning Script

This script fine-tunes a LLaMA 3.2 model on a dataset of folk stories and related questions.
It includes semantic search for context retrieval, LoRA fine-tuning, and model evaluation.

Key components:
- Environment setup and authentication
- Text chunking and semantic search
- Model loading and configuration
- Data preparation and formatting
- LoRA fine-tuning
- Model evaluation using BLEU, ROUGE, and BERTScore
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from langchain.text_splitter import RecursiveCharacterTextSplitter
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Load environment variables from .env file
load_dotenv()
# Get the Hugging Face token from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Check if the environment variable is set
if HUGGINGFACE_TOKEN is None:
    raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable.")

warnings.filterwarnings("ignore")

login(token=HUGGINGFACE_TOKEN)

# Load the embedding model for semantic search
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split a text into smaller chunks for more effective processing.

    Args:
        text (str): The input text to be chunked
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks

    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_relevant_chunks(question, chunks, top_k=3):
    """
    Retrieve the most relevant chunks of text for a given question using semantic search.

    Args:
        question (str): The question to find relevant chunks for
        chunks (list): List of text chunks to search through
        top_k (int): Number of most relevant chunks to return

    Returns:
        str: Concatenated relevant chunks
    """
    if not chunks:
        return ""

    # Get embeddings for question and chunks
    question_embedding = embedding_model.encode([question])[0]
    chunk_embeddings = embedding_model.encode(chunks)

    # Calculate similarity
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

    # Get indices of top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Join the most relevant chunks
    relevant_text = "\n\n".join([chunks[i] for i in top_indices])
    return relevant_text


def get_model_and_tokenizer(model_id):
    """
    Load and configure a pre-trained LLM and its tokenizer with quantization.

    Args:
        model_id (str): Hugging Face model identifier

    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer


# ======================================
# MODEL AND DATA LOADING
# ======================================

# Load model
model_id = "meta-llama/Llama-3.2-3B-Instruct"
model, tokenizer = get_model_and_tokenizer(model_id)
output_model = "llama3.2-3B-Fine-tuned-QA"

# Load and preprocess data
df = pd.read_csv("question_answer_pairs_final.csv", delimiter="#")
stories = pd.read_csv("100_stories.csv", delimiter="#")

df["Index"] = df["Index"].astype(int)
merged_df = df.join(stories.iloc[df["Index"]].reset_index(drop=True))
merged_df = merged_df.drop(columns=["Index", "story_length", "source"])

# Create a dictionary to store chunked stories
story_chunks = {}
for idx, row in stories.iterrows():
    story_chunks[row["title"]] = chunk_text(row["story"])

# Split data
train, test = train_test_split(merged_df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
val = val.reset_index(drop=True)


# ======================================
# DATA FORMATTING
# ======================================


def format_chat_template_with_chunks(example):
    """
    Format training and validation examples using the chat template
    with retrieved relevant chunks.

    Args:
        example (dict): Sample with question, answer, and story information

    Returns:
        dict: Formatted example ready for model training
    """
    # Get relevant chunks instead of the full story
    relevant_context = get_relevant_chunks(
        example["Question"], story_chunks.get(example["title"], [example["story"]])
    )

    messages = [
        {
            "role": "system",
            "content": "Answer the question accurately based on the given story context.",
        },
        {
            "role": "user",
            "content": f"Story name: {example['title']}\n\nContext: {relevant_context}\n\nQuestion: {example['Question']}",
        },
        {"role": "assistant", "content": example["Answer"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"text": prompt}


def format_test_template_with_chunks(example):
    """
    Format test examples using the chat template with retrieved relevant chunks.
    Retains the answer for evaluation purposes.

    Args:
        example (dict): Sample with question, answer, and story information

    Returns:
        dict: Formatted example ready for model evaluation
    """
    # Get relevant chunks instead of the full story
    relevant_context = get_relevant_chunks(
        example["Question"], story_chunks.get(example["title"], [example["story"]])
    )

    messages = [
        {
            "role": "system",
            "content": "Answer the question accurately based on the given story context.",
        },
        {
            "role": "user",
            "content": f"Answer the question given below only using the provided context.\nStory name: {example['title']}\n\nContext: {relevant_context}\n\nQuestion: {example['Question']}",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"text": prompt, "Answer": example["Answer"]}


# Create datasets
train_data = Dataset.from_pandas(train)
val_data = Dataset.from_pandas(val)
test_data = Dataset.from_pandas(test)

train_data = train_data.map(
    format_chat_template_with_chunks, remove_columns=train_data.column_names
)
val_data = val_data.map(
    format_chat_template_with_chunks, remove_columns=val_data.column_names
)
test_data = test_data.map(format_test_template_with_chunks)

# Print sample
print(train_data[0]["text"])

# ======================================
# MODEL TRAINING CONFIGURATION
# ======================================

# Configure LoRA
peft_config = LoraConfig(
    r=16,  # Rank dimension
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",  # Add bias to attention modules
    task_type="CAUSAL_LM",  # The task type (causal language modeling)
)

training_args = TrainingArguments(
    output_dir=output_model,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=10,
    logging_steps=10,
    warmup_steps=10,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config,
    args=training_args,
)

# ======================================
# TRAINING AND MODEL SAVING
# ======================================

# Train the model
trainer.train()

# Save the model
model.save_pretrained(output_model)
tokenizer.save_pretrained(output_model)


# ======================================
# MODEL EVALUATION
# ======================================


def generate_predictions(test_data):
    """
    Generate predictions for the test dataset.

    Args:
        test_data (Dataset): Dataset of test samples

    Returns:
        tuple: (predictions, references) - Lists of model predictions and ground truth answers
    """
    model.eval()
    predictions = []
    references = []

    for example in test_data:
        input_text = example["text"]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
            model.device
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=0.7,  # Controls randomness (lower = less random)
                top_p=0.9,  # Nucleus sampling parameter
                do_sample=True,  # Enable sampling
            )

        # Extract only the generated part (not the input)
        generated_text = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        predictions.append(generated_text.strip().lower())
        references.append(example["Answer"].lower())

    return predictions, references


# Evaluation metrics
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from torchmetrics.functional.text import bleu_score


def evaluate_model(model, test_data):
    """
    Evaluate the fine-tuned model using multiple metrics.

    Args:
        model: The fine-tuned model
        test_data (Dataset): Dataset of test samples

    Returns:
        None: Prints evaluation results
    """
    model.eval()
    predictions, references = generate_predictions(test_data)

    # BLEU Score - Measures n-gram precision
    bleu = bleu_score(predictions, [[ref] for ref in references])
    print(f"BLEU Score: {bleu:.4f}")

    # ROUGE Scores - Measures overlap of n-grams, word sequences, and word pairs
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(pred, ref) for pred, ref in zip(predictions, references)
    ]
    avg_rouge = {
        "rouge1": sum([s["rouge1"].fmeasure for s in rouge_scores]) / len(rouge_scores),
        "rouge2": sum([s["rouge2"].fmeasure for s in rouge_scores]) / len(rouge_scores),
        "rougeL": sum([s["rougeL"].fmeasure for s in rouge_scores]) / len(rouge_scores),
    }
    print(
        f"ROUGE-1: {avg_rouge['rouge1']:.4f}, ROUGE-2: {avg_rouge['rouge2']:.4f}, ROUGE-L: {avg_rouge['rougeL']:.4f}"
    )

    # BERTScore - Contextual embedding-based evaluation metric
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    print(f"BERTScore (F1): {F1.mean():.4f}")


# Evaluate the model
evaluate_model(model, test_data)
