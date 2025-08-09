# GPT-2 Epicurious Recipe Generation (TensorFlow)
# SciNet DAT112: Neural Network Programming
# Adapted by Layal Jbara
#
# This script loads a fine-tuned GPT-2 model and generates a recipe
# based on a given list of ingredients or partial recipe text.
#######################################################################

"""
Generate.py

This script uses a trained GPT-2 model to generate text continuations
for recipe prompts. The workflow is:

1. Load the fine-tuned model from the `output` directory.
2. Load the GPT-2 tokenizer.
3. Encode an input recipe prompt into token IDs.
4. Use the model's `generate()` method to produce text.
5. Decode and print the generated recipe text.

The input prompt in this example contains a recipe title, ingredient
list, and partial preparation steps.
"""

#######################################################################
# Imports
#######################################################################

import transformers

#######################################################################
# Load fine-tuned GPT-2 model
#######################################################################
model = transformers.TFAutoModelForCausalLM.from_pretrained('output')

#######################################################################
# Load tokenizer
#######################################################################
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

#######################################################################
# Encode input recipe prompt
#######################################################################
encoded = tokenizer.encode(
    "Spinach Salad with Warm Feta Dressing\n"
    "1 9-ounce bag fresh spinach leaves\n"
    "5 tablespoons olive oil, divided\n"
    "1 medium red onion, halved, cut into 1/3-inch-thick wedges with some core attached\n"
    "1 7-ounce package feta cheese, coarsely crumbled",
    return_tensors='tf'
)

#######################################################################
# Generate recipe text
#######################################################################
generated = model.generate(encoded, max_length=256)

#######################################################################
# Decode and display output
#######################################################################
print(tokenizer.decode(generated[0]))












ChatGPT can make mistakes. Check important info. See Cookie Preferences.