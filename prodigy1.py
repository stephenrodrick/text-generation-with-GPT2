import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling


def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

def data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  
    )


def fine_tune_gpt2(model_name, train_file_path, output_dir, num_train_epochs=3, batch_size=4):
    
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

    
    train_dataset = load_dataset(train_file_path, tokenizer)

   
    training_args = TrainingArguments(
        output_dir=output_dir,  
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,  
        save_total_limit=2,  
    )

 
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator(tokenizer),
        train_dataset=train_dataset,
    )

    
    trainer.train()

    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved at {output_dir}.")

    return model, tokenizer


def generate_text(prompt, model, tokenizer, max_length=100, num_return_sequences=1):
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  
        top_k=50,  
        top_p=0.95,  
        temperature=0.7,  
    )
    
  
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


if __name__ == "__main__":
    
    training_file = "E:\\biotech.txt"  

   
    model_name = "gpt2"  
    output_dir = "./fine_tuned_gpt2_model"

    
    model, tokenizer = fine_tune_gpt2(
        model_name=model_name,
        train_file_path=training_file,
        output_dir=output_dir,
        num_train_epochs=3,  
        batch_size=4  
    )

    sample_prompt = "The future of AI in biotechnology is"
    generated_texts = generate_text(prompt=sample_prompt, model=model, tokenizer=tokenizer)

    for idx, text in enumerate(generated_texts):
        print(f"Generated Text {idx + 1}:\n{text}\n")
