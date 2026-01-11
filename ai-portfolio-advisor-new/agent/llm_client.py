import json
from groq import Groq

class LLMClient:
    """
    Real LLM Client using Groq API.
    This class handles the communication with Llama-3 models to transform 
    natural language into structured tool calls (JSON).
    """

    def __init__(self):
        # Replace with your actual Groq API Key
        self.api_key = "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" 
        self.client = Groq(api_key=self.api_key)
        
        # Using Llama-3.3 70B for high-reasoning capabilities
        self.model = "llama-3.3-70b-versatile" 

    def chat(self, messages):
        """
        Sends the conversation history to Groq and receives a structured JSON response.
        """
        print(f"\n--- CALLING GROQ AI (Reasoning Engine) ---")
        
        try:
            # API Call to Groq
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for precise and deterministic output
                response_format={"type": "json_object"}  # Enforces JSON output format
            )
            
            # Extract the content from the AI response
            response_content = completion.choices[0].message.content
            
            # Log the raw JSON for debugging purposes
            print(f"DEBUG GROQ RESPONSE: {response_content}")
            
            return response_content

        except Exception as e:
            # Fallback logic in case of connection or API errors
            error_response = {
                "thought": f"Groq API Error: {str(e)}",
                "tool": None,
                "args": {}
            }
            return json.dumps(error_response)
        

    def chat_text(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 500) -> str:
        """
        Returns plain text (no JSON enforcement).
        Used for report commentary.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            text = completion.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            # Keep demo clean: return a short, non-blocking message
            return f"- AI commentary unavailable (Groq error: {str(e)})"
