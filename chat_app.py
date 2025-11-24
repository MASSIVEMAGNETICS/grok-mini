#!/usr/bin/env python3
"""
Grok-Mini Chat - Windows GUI Application
A human-friendly ChatGPT-type interface for Grok-Mini V2
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
from datetime import datetime
import os
import sys

try:
    from grok_mini import GrokMiniV2, generate, config, tokenizer
    import torch
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


class GrokMiniChat:
    def __init__(self, root):
        self.root = root
        self.root.title("Grok-Mini Chat")
        self.root.geometry("900x700")
        
        # Set icon and style
        self.setup_styles()
        
        # Model state
        self.model = None
        self.model_loaded = False
        self.generation_queue = queue.Queue()
        self.is_generating = False
        
        # Chat history
        self.chat_history = []
        
        # Create UI
        self.create_ui()
        
        # Start model loading in background
        self.load_model_async()
    
    def setup_styles(self):
        """Setup UI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.user_msg_color = "#2b5797"
        self.ai_msg_color = "#2d2d2d"
        self.input_bg = "#252525"
        self.button_color = "#0e639c"
        
    def create_ui(self):
        """Create the main UI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar
        title_frame = tk.Frame(main_frame, bg=self.bg_color)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="🤖 Grok-Mini Chat",
            font=("Segoe UI", 20, "bold"),
            bg=self.bg_color,
            fg=self.fg_color
        )
        title_label.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(
            title_frame,
            text="Loading model...",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg="#ffaa00"
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # Chat display area
        chat_frame = tk.Frame(main_frame, bg=self.bg_color)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure tags for message styling
        self.chat_display.tag_config("user", foreground="#7eb6ff", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("ai", foreground="#00ff88", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("system", foreground="#ffaa00", font=("Segoe UI", 9, "italic"))
        self.chat_display.tag_config("timestamp", foreground="#888888", font=("Segoe UI", 8))
        
        # Input area
        input_frame = tk.Frame(main_frame, bg=self.bg_color)
        input_frame.pack(fill=tk.X)
        
        # Input text box
        self.input_text = tk.Text(
            input_frame,
            height=3,
            font=("Segoe UI", 10),
            bg=self.input_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Return>", self.on_enter_key)
        self.input_text.bind("<Shift-Return>", lambda e: None)  # Allow newline with Shift+Enter
        
        # Button frame
        button_frame = tk.Frame(input_frame, bg=self.bg_color)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.send_button = tk.Button(
            button_frame,
            text="Send",
            command=self.send_message,
            font=("Segoe UI", 10, "bold"),
            bg=self.button_color,
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.send_button.pack(pady=(0, 5))
        
        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_chat,
            font=("Segoe UI", 9),
            bg="#4a4a4a",
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=20,
            pady=5,
            cursor="hand2"
        )
        clear_button.pack()
        
        # Settings bar
        settings_frame = tk.Frame(main_frame, bg=self.bg_color)
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            settings_frame,
            text="Temperature:",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_scale = tk.Scale(
            settings_frame,
            from_=0.1,
            to=1.5,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.temp_var,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            length=150
        )
        temp_scale.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            settings_frame,
            text="Max Tokens:",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.max_tokens_var = tk.IntVar(value=150)
        max_tokens_spin = tk.Spinbox(
            settings_frame,
            from_=50,
            to=500,
            textvariable=self.max_tokens_var,
            width=8,
            font=("Segoe UI", 9),
            bg=self.input_bg,
            fg=self.fg_color,
            relief=tk.FLAT
        )
        max_tokens_spin.pack(side=tk.LEFT, padx=(0, 20))
        
        # Load image button
        load_img_button = tk.Button(
            settings_frame,
            text="📷 Load Image",
            command=self.load_image,
            font=("Segoe UI", 9),
            bg="#4a4a4a",
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        load_img_button.pack(side=tk.LEFT)
        
        self.image_path = None
        self.image_label = tk.Label(
            settings_frame,
            text="",
            bg=self.bg_color,
            fg="#00ff88",
            font=("Segoe UI", 8)
        )
        self.image_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Welcome message
        self.add_system_message("Welcome to Grok-Mini Chat! Loading model, please wait...")
    
    def load_model_async(self):
        """Load model in background thread"""
        def load():
            try:
                self.add_system_message(f"Initializing model on {config.device}...")
                self.model = GrokMiniV2().to(config.device).to(config.dtype)
                param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
                self.model_loaded = True
                self.root.after(0, lambda: self.on_model_loaded(param_count))
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_model_loaded(self, param_count):
        """Called when model is loaded successfully"""
        self.status_label.config(text=f"✓ Ready ({param_count:.1f}M params)", fg="#00ff88")
        self.add_system_message(f"Model loaded successfully! ({param_count:.1f}M parameters)")
        self.add_system_message("You can start chatting now. Type your message and press Send or Enter.")
        self.input_text.focus()
    
    def on_model_error(self, error):
        """Called when model loading fails"""
        self.status_label.config(text="⚠ Error loading model", fg="#ff0000")
        self.add_system_message(f"Error loading model: {error}")
        messagebox.showerror("Model Error", f"Failed to load model:\n{error}")
    
    def add_message(self, role, message, tag=None):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # Add role
        role_tag = tag if tag else role.lower()
        self.chat_display.insert(tk.END, f"{role}: ", role_tag)
        
        # Add message
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def add_system_message(self, message):
        """Add a system message"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.chat_display.insert(tk.END, f"{message}\n", "system")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def on_enter_key(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Check if Shift is not pressed
            self.send_message()
            return "break"  # Prevent default newline
    
    def send_message(self):
        """Send user message and get AI response"""
        if not self.model_loaded:
            messagebox.showwarning("Model Not Ready", "Please wait for the model to finish loading.")
            return
        
        if self.is_generating:
            messagebox.showinfo("Generating", "Please wait for the current response to complete.")
            return
        
        # Get user input
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return
        
        # Clear input
        self.input_text.delete("1.0", tk.END)
        
        # Add user message to chat
        self.add_message("You", user_input, "user")
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Disable send button
        self.send_button.config(state=tk.DISABLED, text="Thinking...")
        self.is_generating = True
        
        # Generate response in background
        def generate_response():
            try:
                # Prepare image if loaded
                image_tensor = None
                if self.image_path:
                    try:
                        from PIL import Image
                        import numpy as np
                        img = Image.open(self.image_path).resize((224, 224))
                        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(img_array).unsqueeze(0).to(config.device).to(config.dtype)
                    except Exception as e:
                        print(f"Error loading image: {e}")
                
                # Generate
                response = generate(
                    self.model,
                    user_input,
                    max_new_tokens=self.max_tokens_var.get(),
                    temperature=self.temp_var.get(),
                    image=image_tensor
                )
                
                # Extract only the new part (after the prompt)
                ai_response = response[len(user_input):].strip()
                
                # Update UI in main thread
                self.root.after(0, lambda: self.on_response_generated(ai_response))
                
            except Exception as e:
                self.root.after(0, lambda: self.on_generation_error(str(e)))
        
        thread = threading.Thread(target=generate_response, daemon=True)
        thread.start()
    
    def on_response_generated(self, response):
        """Called when response is generated"""
        self.add_message("Grok-Mini", response, "ai")
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Re-enable send button
        self.send_button.config(state=tk.NORMAL, text="Send")
        self.is_generating = False
        
        # Clear image after use
        if self.image_path:
            self.image_path = None
            self.image_label.config(text="")
    
    def on_generation_error(self, error):
        """Called when generation fails"""
        self.add_system_message(f"Error generating response: {error}")
        self.send_button.config(state=tk.NORMAL, text="Send")
        self.is_generating = False
    
    def clear_chat(self):
        """Clear chat history"""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat history?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.chat_history = []
            self.add_system_message("Chat cleared. Start a new conversation!")
    
    def load_image(self):
        """Load an image for vision input"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.image_path = file_path
            filename = os.path.basename(file_path)
            self.image_label.config(text=f"📷 {filename}")
            self.add_system_message(f"Image loaded: {filename}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = GrokMiniChat(root)
    root.mainloop()


if __name__ == "__main__":
    main()
