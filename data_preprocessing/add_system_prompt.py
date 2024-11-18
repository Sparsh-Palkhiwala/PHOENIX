import json

# Load existing JSON data
with open("C:/Users/ASUS/ghq/github.com/Sparsh-Palkhiwala/PHOENIX/Output_json/Base_output.json", "r") as f:
    json_data = json.load(f)

# Define the system prompt
system_prompt = (
    "You are an advanced reasoning assistant specializing in logical problem-solving. "
    "Your role is to carefully analyze the provided context and query, "
    "incorporate the reasoning framework represented by the graph and depth, try to understand the reasoning process, "
    "and generate a precise and well-reasoned answer. Every answer should be a step by step reasoning process. "
    "Always express reasoning of probability with . "
    "Always prioritize logical consistency and provide clear, step-by-step reasoning "
    "when formulating your response."
    
)

# Encapsulate the data with the system prompt
final_data = {
    "system_prompt": system_prompt,
    "data": json_data
}

# Save the updated JSON
with open("C:/Users/ASUS/ghq/github.com/Sparsh-Palkhiwala/PHOENIX/Output_json/Base_output_with_system_prompt.json", "w") as f:
    json.dump(final_data, f, indent=4)
