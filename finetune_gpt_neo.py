from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings

# Load the dataset
dataset = load_dataset("text", data_files={"train": "data/formatted_cladder.txt"})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to False for causal language modeling
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory to save the model
    evaluation_strategy="no",         # Disable evaluation
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=8,   # Batch size per device
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Weight decay for optimization
    save_strategy="epoch",           # Save model after each epoch
    logging_dir="./logs",            # Directory for logging
    logging_steps=500,               # Log training progress every 500 steps
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

