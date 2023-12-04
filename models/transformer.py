from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def hello_world():"

tokens = tokenizer(text, return_tensors="pt")

print(tokenizer(text, return_tensors="pt"))

model.eval()
completion = model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

print(tokenizer.decode(completion[0]))


# Creating the trace
traced_model = torch.jit.trace(model, [tokens['input_ids'], tokens['attention_mask']])
#torch.jit.save(traced_model, "traced_bert.pt")
