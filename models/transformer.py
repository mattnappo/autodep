from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
model = GPT2LMHeadModel.from_pretrained("facebook/opt-125m")

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Save the TorchScript model
scripted_model.save("opt-125m.pt")
