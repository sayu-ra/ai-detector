import tkinter as tk
from tkinter import scrolledtext
import re

# Standardize colors
RED_HIGHLIGHT = "#FFCCCC" 

class DashboardUI:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.root.title("DeepGuard AI Suite")
        self.root.geometry("1100x700")
        self.root.configure(bg="#F8F9FA")
        
        self.setup_sidebar()
        self.main_container = tk.Frame(self.root, bg="#F8F9FA")
        self.main_container.pack(side="right", expand=True, fill="both")
        self.show_text_interface()

    def setup_sidebar(self):
        sidebar = tk.Frame(self.root, bg="#1A202C", width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="DEEPGUARD", fg="white", bg="#1A202C", font=("Helvetica", 16, "bold")).pack(pady=30)

    def show_text_interface(self):
        tk.Label(self.main_container, text="AI Text Analysis", font=("Helvetica", 20, "bold"), bg="#F8F9FA").pack(anchor="w", padx=40, pady=40)
        
        content = tk.Frame(self.main_container, bg="#F8F9FA")
        content.pack(fill="both", expand=True, padx=40)

        self.input_area = scrolledtext.ScrolledText(content, font=("Helvetica", 12), relief="flat", padx=15, pady=15)
        self.input_area.pack(side="left", fill="both", expand=True)
        self.input_area.tag_config("ai_warning", background=RED_HIGHLIGHT, foreground="black")

        self.card = tk.Frame(content, bg="white", width=280)
        self.card.pack(side="right", fill="y", padx=(20, 0))
        self.card.pack_propagate(False)
        self.score_display = tk.Label(self.card, text="0%", font=("Helvetica", 48, "bold"), bg="white")
        self.score_display.pack(pady=(60, 10))

        tk.Button(self.main_container, text="Run Analysis", command=self.on_analyze_click, bg="#E6A23C", fg="white", font=("Helvetica", 12, "bold"), padx=40, pady=12).pack(pady=30)

    def on_analyze_click(self):
        # Clear previous highlights
        self.input_area.tag_remove("ai_warning", "1.0", tk.END)
        
        text = self.input_area.get("1.0", tk.END).strip()
        if not text: return

        # Get the average score and the specific list of high-prob sentences
        avg_score, flagged_list = self.controller.process_text_request(text)
        
        # Update sidebar
        self.score_display.config(text=f"{avg_score}%", 
                                  fg="#E53E3E" if avg_score > 50 else "#2D3748")
        
        # Highlight ONLY the sentences that were individually high
        if flagged_list:
            for sentence in flagged_list:
                self.apply_robust_highlight(sentence)

    def apply_robust_highlight(self, sentence):
        """Finds the exact sentence in the text widget and applies the red tag."""
        import re
        full_text = self.input_area.get("1.0", tk.END)
        
        # Use regex to find the sentence even if there are weird spaces
        pattern = re.escape(sentence).replace(r'\ ', r'\s+')
        for match in re.finditer(pattern, full_text):
            # Convert raw character index to Tkinter line.col format
            start_idx = self.get_tk_index(full_text, match.start())
            end_idx = self.get_tk_index(full_text, match.end())
            self.input_area.tag_add("ai_warning", start_idx, end_idx)

    def get_tk_index(self, text, index):
        line = text.count('\n', 0, index) + 1
        col = index - text.rfind('\n', 0, index) - 1
        return f"{line}.{col}"