from transformers import AutoTokenizer



# ********DOWNLOADS GPT2 TOKENIZER AND ADDS REQUIRED SPECIAL TOKENS *********
# This tokenizer is used for all models going forward

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define special tokens
token_nontoxic = '<|nontoxic|>'
token_toxic = '<|toxic|>'
token_pos = '<|pos|>'
token_neu = '<|neu|>'
token_neg = '<|neg|>'

# add the pad token:
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# add the special tokens:
special_tokens_dict = {'additional_special_tokens': [
    token_nontoxic,
    token_toxic,
    token_pos,
    token_neu,
    token_neg]
}
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.save_pretrained('tokenizer')