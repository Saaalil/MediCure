import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
from langchain import LLMChain, PromptTemplate
from langchain_ollama import OllamaLLM
from rich.console import Console

# Initialize the console for pretty printing
console = Console()

# Initialize the LLM with the appropriate model
llm = OllamaLLM(model='llama3.1', base_url='http://localhost:11434')

# Load the pre-trained VGG-16 model
device = "cuda" if torch.cuda.is_available() else "cpu"
vgg16_model = models.vgg16(pretrained=True).to(device)
vgg16_model.eval()

# Image transformations for VGG-16 input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cache file location
CACHE_FILE = "cache.json"

# -------------------- Caching Functions --------------------
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
    return diagnosis

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
# -------------------- End of Caching Functions --------------------

# -------------------- MRI Image Classification Functions --------------------
def classify_mri_image(image_path):
    """
    Classify the MRI image using VGG-16 and return the predicted label.
    """
    # Open image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Perform classification
    with torch.no_grad():
        outputs = vgg16_model(image)
        _, predicted = torch.max(outputs, 1)
        
    # Map the predicted label to the corresponding class
    class_names = ["class_1", "class_2", "class_3"]  # You will need to customize these
    if predicted.item() >= len(class_names):
        return "Unknown"
    else:
        predicted_class = class_names[predicted.item()]
        return predicted_class

# -------------------- LLM Processing --------------------
# Define the multimodal prompt template
prompt = PromptTemplate(
    input_variables=["user_input", "mri_classification", "image_features"],
    template="""You are an AI assistant designed to help diagnose potential health issues based on reported symptoms and MRI image analysis.
    You are provided with text describing symptoms and an MRI scan classification. 
    Based on the MRI classification: {mri_classification} and user input: {user_input}, analyze the potential diagnosis and suggest severity levels.
    The most likely diagnosis based on the symptoms is: [DIAGNOSIS]
    The severity level is: [RED/ORANGE/YELLOW/BLUE/GREEN]
    [RECOMMENDED ACTIONS BASED ON SEVERITY LEVEL]
    Severity Level Guidelines:
    RED (80-100): Rush to the hospital immediately.
    ORANGE (60-80): Consult a doctor soon and follow basic remedies until then.
    YELLOW (40-60): Visit a Clinic or take an online consultation.
    BLUE (20-40): Mild issue. Suggest home remedies.
    GREEN (0-20): No significant health problem, no medical visit required.
    """
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

def generate_multimodal_prompt(mri_classification, user_input):
    """
    Combines text and MRI classification into a single prompt
    """
    combined_input = f"User input: {user_input}\nMRI Classification: {mri_classification}"
    return combined_input

def handle_input(user_input, image_path):
    """
    Processes the user input, interacts with the LLM, and provides recommendations based on text and MRI image.
    """
    # Classify the MRI image
    mri_classification = classify_mri_image(image_path)

    # Create the multimodal prompt
    multimodal_prompt = generate_multimodal_prompt(mri_classification, user_input)

    # Load the cache
    cache = load_cache()

    # Normalize the input
    normalized_input = user_input.lower().strip()

    # Check if the diagnosis exists in the cache
    cached_response = get_cached_response(normalized_input, cache)
    if cached_response:
        console.print(f"Assistant (from cache): {cached_response}")
    else:
        try:
            # Generate response using the LLM
            assistant_response = llm_chain.predict(user_input=user_input, mri_classification=mri_classification, image_features=None)
            # Normalize and cache the response
            diagnosis = assistant_response.split('\n')[0].strip()  # Assuming diagnosis is on the first line
            normalized_diagnosis = normalize_diagnosis(diagnosis)
            cache_response(normalized_diagnosis, assistant_response, cache)
            console.print(f"Assistant: {assistant_response}")
        except Exception as e:
            console.print(f"Error generating response: {e}", style="bold red")
            console.print("Assistant: Unable to process the input at this time.")

# -------------------- GUI Functions --------------------
def run_app():
    """
    Runs the main application loop, handling user input and image selection.
    """
    # Open the file dialog and get the selected image path
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select MRI Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")])
    
    if image_path:
        console.print("Assistant: Please describe your symptoms.")
        user_input = input("User: ")
        handle_input(user_input, image_path)
    else:
        console.print("Assistant: No image selected. Please try again.")

# Main Function
if __name__ == "__main__":
    run_app()