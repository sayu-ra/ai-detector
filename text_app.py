import tkinter as tk
import threading
import nltk
import os
from ui_components import DashboardUI
from inference import Detector

class AIController:
    def __init__(self):
        self.root = tk.Tk()
        # FIX: Ensure this matches the ui_components __init__
        self.ui = DashboardUI(self.root, self)
        
        self.text_engine = None
        threading.Thread(target=self.load_engines, daemon=True).start()

    def load_engines(self):
        try:
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except:
                nltk.download('punkt')
            
            model_path = "./model_assets/text_detector"
            if os.path.exists(model_path):
                self.text_engine = Detector(model_path)
                print("Text Engine Loaded Successfully.")
        except Exception as e:
            print(f"Engine Load Error: {e}")

    def process_text_request(self, text):
        if not self.text_engine or not text.strip():
            return 0, []

        sentences = nltk.sent_tokenize(text.replace('\n', ' '))
        results = []

        for s in sentences:
            clean_s = s.strip()
            if len(clean_s.split()) < 4: continue
                
            prob = self.text_engine.predict(clean_s)
            results.append({"text": clean_s, "score": prob})

        if not results: return 0, []

        # Calculate average
        avg_overall = round(sum(r['score'] for r in results) / len(results))
        
        # DYNAMIC THRESHOLD: Find the highest score in this specific text
        max_score = max(r['score'] for r in results)
        
        # Highlight anything that is within 10% of the highest score, 
        # OR anything above 35% (since your model is giving lower scores)
        threshold = max(35, max_score - 10)
        
        flagged = [r['text'] for r in results if r['score'] >= threshold]

        print(f"DEBUG: Max={max_score}%, Threshold={threshold}%, Highlighting={len(flagged)}")
        return avg_overall, flagged

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    AIController().run()