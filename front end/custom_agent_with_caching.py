import re
import requests
import json  # Added for handling JSON operations
from rich.console import Console
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM
import asyncio
# Initialize the console for pretty printing
console = Console()

# Initialize the LLM with the appropriate model
# Ensure that the model name matches the one you have access to
llm = OllamaLLM(model='llama3.1', base_url='http://localhost:11434')

# Define the updated prompt template
prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an AI assistant designed to help diagnose potential health issues based on reported symptoms.
Your goal is to analyze the provided symptoms and suggest possible diagnoses along with severity levels.
and mainly be ready with Factual Answers and Questions Based on the symptoms: {user_input}, please follow these steps:
1. Read through the symptoms carefully and think through potential diagnoses.
2. Provide your assessment in the following format:
The most likely diagnosis based on the symptoms is: [DIAGNOSIS]
The severity level is: [RED/ORANGE/YELLOW/BLUE/GREEN]
[RECOMMENDED ACTIONS BASED ON SEVERITY LEVEL GUIDELINES PROVIDED BELOW]
Severity Level Guidelines:
RED (80-100): Rush to the hospital immediately.
ORANGE (60-80): Consult a doctor soon and follow basic remedies until then.
YELLOW (40-60): Visit a Clinic or take an online consultation.
BLUE (20-40): Mild issue. Suggest home remedies.
GREEN (0-20): No significant health problem, no medical visit required.
3. If no diagnosis can be made based on the symptoms, state that clearly.
"""
)

# Create the LLMChain with the prompt and LLM
llm_chain = LLMChain(llm=llm, prompt=prompt)

# -------------------- Caching Functionality Starts Here --------------------

CACHE_FILE = r"C:\Users\SALIL HIREMATH\Desktop\iic hacathon\cache.json"  # Define the cache file path


pipeline = prompt | llm

import asyncio

# Assume that 'pipeline.invoke' is a synchronous function. 
# We'll handle it as a blocking operation and run it in an async-friendly way.

async def generate_response(user_input):
    """
    Generates a response using the LLM pipeline asynchronously.
    """
    try:
        # If pipeline.invoke is synchronous, use asyncio.to_thread to run it in the background
        response = await asyncio.to_thread(pipeline.invoke, {"user_input": user_input})
        return response
    except Exception as e:
        console.print(f"Error generating response: {e}", style="bold red")
        return "An error occurred. Please try again later."


def load_cache():
    """
    Loads the cache from the JSON file.
    Returns an empty dictionary if the file does not exist or if there's a JSONDecodeError.
    """
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    return cache

def save_cache(cache):
    """
    Saves the cache dictionary to the JSON file.
    """
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def normalize_diagnosis(diagnosis):
    """
    Normalizes the diagnosis string to ensure consistent cache keys.
    """
    diagnosis = diagnosis.lower().strip()
    # Remove any trailing asterisks or special characters
    diagnosis = diagnosis.rstrip('*').strip()
    
    # Define a mapping for common synonyms or variations
    diagnosis_mapping = {
        "viral upper respiratory tract infection (common cold)": "common cold",
        "common cold": "common cold",
        "urti (upper respiratory tract infection)": "common cold",
        "asthma attack": "asthma attack",
        # Add more mappings as needed
    }
    
    # Return the normalized diagnosis if it exists in the mapping
    return diagnosis_mapping.get(diagnosis, diagnosis)  # Default to the original if not mapped

def get_cached_response(diagnosis, cache):
    """
    Retrieves the cached response for a given diagnosis.
    Returns the response if found, else None.
    """
    return cache.get(diagnosis)

def cache_response(diagnosis, assistant_response, cache):
    """
    Caches the assistant response using the normalized diagnosis as the key.
    """
    cache[diagnosis] = assistant_response
    save_cache(cache)

def extract_diagnosis(response_text):
    """
    Extracts the diagnosis from the assistant's response using regex.
    """
    match = re.search(r'The most likely diagnosis based on the symptoms is:\s*(.*)', response_text, re.IGNORECASE)
    if match:
        diagnosis = match.group(1).strip()
        # Remove any trailing lines or extra information
        diagnosis = diagnosis.split('\n')[0]
        return diagnosis
    else:
        return None

def extract_severity_level(response_text):
    """
    Extracts the severity level from the assistant's response using regex.
    """
    match = re.search(r'The severity level is:\s*(\w+)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        return None

# -------------------- Caching Functionality Ends Here --------------------

def handle_input(user_input):
    """
    Processes the user input, interacts with the LLM, and provides recommendations.
    Utilizes caching to store and retrieve responses based on diagnosis keywords.
    """
    normalized_input = user_input.lower().strip()

    # Predefined responses for basic greetings and farewells
    basic_responses = {
        "hello": "Hello! How can I assist you with your symptoms today?",
        "hi": "Hi there! Please describe your symptoms, and I'll do my best to help.",
        "thank you": "You're welcome! Take care.",
        "thanks": "Glad I could help!",
        "bye": "Goodbye! Feel free to reach out if you have more questions."
    }

    # Check if the input is a basic response
    if normalized_input in basic_responses:
        response = basic_responses[normalized_input]
        console.print(f"Assistant: {response}")
        return
    else:
        # Proceed with LLM processing for other inputs
        # Load the cache
        cache = load_cache()

        # Prepare the LLM input
        llm_input = {
            "user_input": user_input
        }

        # Get the assistant's response from the LLM
        try:
            assistant_response = llm_chain.predict(**llm_input)
        except Exception as e:
            console.print(f"Assistant: I'm sorry, but I'm experiencing some issues right now. Please try again later.", style="bold red")
            console.print(f"Error Details: {e}", style="dim")
            return

        # Process the response as usual
        diagnosis = extract_diagnosis(assistant_response)
        if diagnosis:
            normalized_diagnosis = normalize_diagnosis(diagnosis)
            cached_response = get_cached_response(normalized_diagnosis, cache)
            if cached_response:
                console.print(f"Assistant: {cached_response}")
            else:
                cache_response(normalized_diagnosis, assistant_response, cache)
                console.print(f"Assistant: {assistant_response}")
        else:
            console.print(f"Assistant: {assistant_response}")

        severity_level = extract_severity_level(assistant_response)
        if severity_level:
            recommendation = get_recommendation(severity_level)
            console.print(f"Recommendation: {recommendation}")
        else:
            console.print("Recommendation: Please consult a healthcare professional for an accurate diagnosis and appropriate treatment.")


def get_recommendation(severity_level):
    """
    Provides a recommendation based on the severity level.
    """
    severity_level = severity_level.upper()
    if severity_level == "RED":
        return "Rush to the hospital immediately."
    elif severity_level == "ORANGE":
        return "Consult a doctor soon and follow basic remedies until then."
    elif severity_level == "YELLOW":
        return "Visit a Clinic or take an online consultation."
    elif severity_level == "BLUE":
        return "Mild issue. Suggest home remedies."
    elif severity_level == "GREEN":
        return "No significant health problem, no medical visit required."
    else:
        return "Please consult a healthcare professional for an accurate diagnosis and appropriate treatment."

if __name__ == "__main__":
    console.print("Assistant: Hi there! Please describe your symptoms, and I'll do my best to help.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            console.print("Assistant: Goodbye! Take care.")
            break
        handle_input(user_input)
