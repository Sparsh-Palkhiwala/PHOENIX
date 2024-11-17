import pandas as pd
import json

# Load the dataset
data = pd.read_csv("C:/Users/ASUS/ghq/github.com/Sparsh-Palkhiwala/PHOENIX/BLInD/datasets/Base_1000_examples.csv")

# Function to create JSON structure
def row_to_json(row):
    return {
        "input": f"Context: {row['contexts']}. Query: {row['query']}.",
        "reasoning": {
            "graph": row['graph'],
            "depth": row['depth']
        },
        "output": row['answers']
    }

# Apply the function to all rows
json_data = data.apply(row_to_json, axis=1).tolist()

# Save to a JSON file
with open("C:/Users/ASUS/ghq/github.com/Sparsh-Palkhiwala/PHOENIX/BLInD/Output_json/Base_output.json", "w") as f:
    json.dump(json_data, f, indent=4)
    print("JSON file has been created successfully.")