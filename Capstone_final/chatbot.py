#!/usr/bin/env python3
"""
chatbot.py

A simple Python chatbot using Microsoft’s DialoGPT model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1) Install requirements:
    #    pip install torch transformers
    #
    # 2) The first run will download the model (~350 MB).
    #
    # 3) Then just:
    #       python chatbot.py
    #    and start chatting! Type "exit" or Ctrl-C to quit.

    model_name = "microsoft/DialoGPT-medium"
    print(f"Loading model {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)

    chat_history_ids = None
    step = 0

    print("\n🤖 DialoGPT Chatbot (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye! 👋")
                break

            # encode the user input and add end-of-string token
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

            # append to chat history (if present)
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if step > 0 else new_input_ids

            # generate a response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=bot_input_ids.shape[-1] + 50,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.75
            )

            # decode the bot’s reply (only the new tokens)
            reply = tokenizer.decode(
                chat_history_ids[0, bot_input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            print(f"Bot: {reply}\n")

            step += 1

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break

if __name__ == "__main__":
    main()
