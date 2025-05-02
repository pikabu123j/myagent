from transformers import pipeline
import re
import torch
# from huggingface_hub import login
# login(token="read")  # Add this before pipeline()

class LLMAgents:
    def __init__(self):
       self.device = "cuda" if torch.cuda.is_available() else "cpu"
       self.analyst = pipeline(
            "text-generation",
            model="gpt2",  # or your preferred model
            device=self.device,
            max_length=100,
            truncation=True,
            pad_token_id=50256  # For GPT-2
        )

        
    def analyze_traffic(self, traffic_data: dict) -> dict:
        prompt = f"""
        Analyze this network traffic data:
        {traffic_data}
        
        Return JSON with:
        - dominant_application (voip|video|data)
        - predicted_demand_change (float)
        """
        response = self.analyst(prompt, max_length=200)
        return self._parse_json(response[0]['generated_text'])
    
    def _parse_json(self, text: str) -> dict:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return eval(match.group()) if match else {}