"""
Question-Answer Pair Generator for Folk Stories

This script generates question-answer pairs from a collection of folk stories using LLM.
It reads stories from a CSV file, processes them through a language model,
and saves the generated QA pairs to output files.

Requirements:
- langchain
- pandas
- re
- json
- dotenv
- Any OpenAI-compatible LLM (e.g., Qwen/QwQ-32B)
"""

import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
# Get the BASE_URL_HERE and API_KEY_HERE from environment variables
BASE_URL_HERE = os.getenv("BASE_URL_HERE")
API_KEY_HERE = os.getenv("API_KEY_HERE")
# Check if the environment variables are set
if BASE_URL_HERE is None or API_KEY_HERE is None:
    raise ValueError(
        "Please set the BASE_URL_HERE and API_KEY_HERE environment variables."
    )


def parse_tuple_format(response_text: str) -> list:
    """
    Parse the response text from LLM to extract question-answer pairs.

    Args:
        response_text (str): The raw response from the language model

    Returns:
        list: A list of dictionaries with 'question' and 'answer' keys
    """
    # Clean up the response text
    cleaned_text = response_text.strip()

    # Use regex to extract question-answer pairs
    pattern = r'\("question": "(.*?)", "answer": "(.*?)"\)'
    matches = re.findall(pattern, cleaned_text)

    # Convert to the desired format
    qa_pairs = [{"question": q, "answer": a} for q, a in matches]

    return qa_pairs


def generate_qa_pairs(text: str, num_pairs: int) -> str:
    """
    Generate question-answer pairs from a given text using a language model.

    Args:
        text (str): The input text (story) to generate questions from
        num_pairs (int): The number of question-answer pairs to generate

    Returns:
        str: The raw response from the language model containing QA pairs

    Raises:
        Exception: If there's an error during generation
    """
    # Initialize the language model
    chat = ChatOpenAI(
        openai_api_base=BASE_URL_HERE,
        openai_api_key=API_KEY_HERE,
        model_name="Qwen/QwQ-32B",
    )

    # Remove any curly braces from the text to avoid formatting issues
    text = text.replace("{", "").replace("}", "")

    # Construct prompt for the language model
    prompt_template = f"""You are an AI that generates {num_pairs} question-answer pairs from the following text:
    
    {text}
    
    Generate concise and relevant questions along with their correct answers. 1 or 2 question can be about theme or moral of the text.
    Provide output in following JSON format as:
    [
        ("question": "First question", "answer": "First answer"),
        ("question": "Second question", "answer": "Second answer"),
        ...
    ]
    Make sure to note that replace the ( and ) with curly braces so that it is properly parsed in JSON format.
    """
    try:
        prompt = prompt_template.format(text=text, num_pairs=num_pairs)
        # Send the prompt to the language model
        response = chat.predict(prompt)
    except Exception as e:
        print(e)
        response = None

    return response


if __name__ == "__main__":
    # Load folk stories from CSV file
    df = pd.read_csv("./data/100_stories.csv", sep="#")
    document = df["story"].tolist()
    title = df["title"].tolist()

    # Configuration parameters
    num_pairs = 15  # Number of QA pairs to generate per story
    INDEX = 0  # Starting index for processing
    document = document[INDEX:]

    # Output file for the final QA pairs
    with open("question_answer_pairs_final.csv", "a") as f:
        for i, doc in enumerate(document):
            text = doc
            print(f"Processing document {i+INDEX}")

            # Generate QA pairs for the current story
            qa_pairs = generate_qa_pairs(text, num_pairs)

            try:
                # Save raw LLM response to a text file
                with open("qa_pairs.txt", "a") as t:
                    t.write(qa_pairs.split("</think>")[1].strip())
                    t.write("\n")

                # Parse the LLM response to extract QA pairs
                qa_pairs = json.loads(qa_pairs.split("</think>")[1].strip())

                # Write the QA pairs to the CSV file
                for idx, pair in enumerate(qa_pairs, 1):
                    f.write(f"{str(i+INDEX)}#{pair['question']}#{pair['answer']}\n")
                f.flush()  # Ensure data is written immediately
            except Exception as e:
                # Log errors for debugging
                print("Index:", i)
                print("Error", e)
