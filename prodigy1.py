import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Function to load the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

# Function to create a data collator for training
def data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Masked Language Model is set to False for GPT-2
    )

# Function to fine-tune the model
def fine_tune_gpt2(model_name, train_file_path, output_dir, num_train_epochs=3, batch_size=4):
    # Load the pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

    # Load the training dataset
    train_dataset = load_dataset(train_file_path, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Directory to store the fine-tuned model
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,  # Save model checkpoint every 500 steps
        save_total_limit=2,  # Keep only the latest 2 models
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator(tokenizer),
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved at {output_dir}.")

    return model, tokenizer

# Function to generate text using the fine-tuned model
def generate_text(prompt, model, tokenizer, max_length=100, num_return_sequences=1):
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text using the model
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  # Prevent repeating the same phrases
        top_k=50,  # Top-k sampling for randomness
        top_p=0.95,  # Nucleus sampling for randomness
        temperature=0.7,  # Adjust the randomness of predictions
    )
    
    # Decode and return the generated text
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Main code to fine-tune and generate text
if __name__ == "__main__":
    # Set your file path to the training text file
    training_file = "E:\\biotech.txt"  # Replace with the path to your dataset file

    # Specify the model name and output directory
    model_name = "gpt2"  # You can also use 'gpt2-medium' or other variants
    output_dir = "./fine_tuned_gpt2_model"

    # Fine-tune the model
    model, tokenizer = fine_tune_gpt2(
        model_name=model_name,
        train_file_path=training_file,
        output_dir=output_dir,
        num_train_epochs=3,  # Number of epochs to train
        batch_size=4  # Adjust based on your system's capability
    )

    # Test the fine-tuned model by generating some text
    sample_prompt = "The future of AI in biotechnology is"
    generated_texts = generate_text(prompt=sample_prompt, model=model, tokenizer=tokenizer)

    # Print the generated text
    for idx, text in enumerate(generated_texts):
        print(f"Generated Text {idx + 1}:\n{text}\n")
