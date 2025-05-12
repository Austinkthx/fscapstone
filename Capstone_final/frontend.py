#!/usr/bin/env python3
"""
chatbot_gui.py

GUI frontend for DialoGPT chatbot using Tkinter
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DialoGPT Chatbot")
        
        # Model configuration
        self.model_name = "microsoft/DialoGPT-medium"
        self.model = None
        self.tokenizer = None
        self.chat_history_ids = None
        self.step = 0
        
        # Initialize model in background
        self.loading = True
        self.init_model()
        
        # Create GUI components
        self.create_widgets()
        
    def create_widgets(self):
        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            width=60,
            height=20,
            state='disabled'
        )
        self.chat_history.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        
        # User input field
        self.user_input = ttk.Entry(self.root, width=50)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind("<Return>", self.send_message)
        
        # Send button
        self.send_button = ttk.Button(
            self.root,
            text="Send",
            command=self.send_message
        )
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        
        # Status bar
        self.status = ttk.Label(self.root, text="Initializing model...")
        self.status.grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
    def init_model(self):
        def load_model():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.loading = False
                self.status.config(text="Ready")
            except Exception as e:
                self.status.config(text=f"Error loading model: {str(e)}")
            
        threading.Thread(target=load_model).start()
        
    def update_chat_history(self, message, is_user=True):
        self.chat_history.configure(state='normal')
        tag = "user" if is_user else "bot"
        self.chat_history.insert(tk.END, f"{'You' if is_user else 'Bot'}: {message}\n", tag)
        self.chat_history.configure(state='disabled')
        self.chat_history.see(tk.END)
        
    def generate_response(self, user_input):
        if self.loading:
            return "Model still loading, please wait..."
            
        try:
            new_input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token,
                return_tensors='pt'
            )
            
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.step > 0 else new_input_ids
            
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=bot_input_ids.shape[-1] + 50,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.75
            )
            
            reply = self.tokenizer.decode(
                self.chat_history_ids[0, bot_input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            self.step += 1
            return reply
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def send_message(self, event=None):
        user_input = self.user_input.get().strip()
        if not user_input:
            return
            
        self.user_input.delete(0, tk.END)
        self.update_chat_history(user_input, is_user=True)
        
        # Disable input while processing
        self.user_input.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.status.config(text="Generating response...")
        
        def process_response():
            response = self.generate_response(user_input)
            self.update_chat_history(response, is_user=False)
            self.user_input.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.status.config(text="Ready")
            
        threading.Thread(target=process_response).start()

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatBotGUI(root)
    
    # Configure tags for message styling
    gui.chat_history.tag_config("user", foreground="blue")
    gui.chat_history.tag_config("bot", foreground="green")
    
    root.mainloop()