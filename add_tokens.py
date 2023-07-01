import transformers
from transformers import T5Tokenizer
import os
# os.makedirs("Tokenizer", exist_ok=True)
tokenizer = T5Tokenizer.from_pretrained("t5-base")
original_num_tokens = len(tokenizer)
extra_tokens = ["<{}>".format(i) for i in range (100)]
num_added_tokens = tokenizer.add_tokens(extra_tokens)
tokenizer.save_pretrained("Tokenizer_Vid2Seq/")