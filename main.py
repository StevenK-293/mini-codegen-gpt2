import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

special_tokens_dict = {"additional_special_tokens": ["<CODE>", "</CODE>"]}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added} special tokens.")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Added [PAD] as pad_token.")

model.resize_token_embeddings(len(tokenizer))

train_texts = [
    "<CODE>\nfunction add(a, b):\n    return a + b\n</CODE>",
    "<CODE>\nfunction greet(name):\n    print('Hello, ' + name)\n</CODE>",
    "<CODE>\nfor i in range(10):\n    print(i)\n</CODE>",
    "<CODE>\nif x > 0:\n    print('positive')\nelse:\n    print('non-positive')\n</CODE>",
    "<CODE>\nfunction factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n</CODE>"
]
dataset = Dataset.from_dict({"text": train_texts})

def tokenz(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_attention_mask=True)

tokenized_dataset = dataset.map(tokenz, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,                
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./codegen_model")
tokenizer.save_pretrained("./codegen_model")

prompt = "<CODE>\nfunction factorial(n):"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)

output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\ngeneratee plz:\n", generated_text)
