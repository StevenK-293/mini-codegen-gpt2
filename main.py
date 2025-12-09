import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import re

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
    "<CODE>\nfunction add(a, b)\n    return a + b\nend\n</CODE>",
    "<CODE>\nfunction greet(name)\n    print('Hello, ' .. name)\nend\n</CODE>",
    "<CODE>\nfor i = 0, 9 do\n    print(i)\nend\n</CODE>",
    "<CODE>\nif x > 0 then\n    print('positive')\nelse\n    print('non-positive')\nend\n</CODE>",
    "<CODE>\nfunction factorial(n)\n    if n <= 1 then\n        return 1\n    else\n        return n * factorial(n - 1)\n    end\nend\n</CODE>",
    "<CODE>\nfunction multiply(x, y)\n    return x * y\nend\n</CODE>",
    "<CODE>\nlocal SimpleClass = {}\nfunction SimpleClass.new(value)\n    local self = setmetatable({}, {__index = SimpleClass})\n    self.value = value\n    return self\nend\nfunction SimpleClass:get_value()\n    return self.value\nend\n</CODE>",
    "<CODE>\nfunction is_even(num)\n    return num % 2 == 0\nend\n</CODE>",
    "<CODE>\nfunction circle_area(radius)\n    return math.pi * radius ^ 2\nend\n</CODE>",
    "<CODE>\nfunction list_sum(lst)\n    local total = 0\n    for _, item in ipairs(lst) do\n        total = total + item\n    end\n    return total\nend\n</CODE>"
]

dataset = Dataset.from_dict({"text": train_texts})

def tokenz(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_attention_mask=True)

tokenized_dataset = dataset.map(tokenz, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=5e-5,
    warmup_steps=10,
    weight_decay=0.01,
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

def extract_code(generated_text):
    match = re.search(r"<CODE>(.*?)</CODE>", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

prompt = "<CODE>\nfunction factorial(n)"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)

output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    no_repeat_ngram_size=2
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
code_str = extract_code(generated_text)

if code_str:
    print("\bruh code:\n", code_str)
else:
    print("\ngenerated text (no code block found):\n", generated_text)
