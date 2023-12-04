'''
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
res = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(res)
'''


from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example input
text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")

# Ensure the model is in evaluation mode and no gradients are calculated
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Creating the trace
traced_model = torch.jit.script(model, (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']), strict=False)

# Save the traced model
traced_model.save("scripted_bert_model.pt")
