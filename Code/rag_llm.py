import os
import json
import time
from typing import Dict, List, Any, Tuple
import config
from Code.template_editor import TemplateEditor
from Code.rag_prompt_builder import SYSTEM_PROMPT_RAG, build_rag_prompt
from Code.prompt_llm import Stage1LLMPipeline

class RAGLLMPipeline:
    """
    Stage 2 RAG LLM Pipeline.
    Constructs dynamic few-shot prompts, queries Gemini API, and applies deterministic template editor.
    Includes rate limit retry logic.
    """
    def __init__(self):
        self.api_key = config.get_api_key()
        self.model_name = config.MODEL_NAME
        self.client = None
        self.fallback_pipeline = Stage1LLMPipeline()
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print(f"[RAG Pipeline] Successfully initialized Gemini API client with model '{self.model_name}'.")
            except Exception as e:
                print(f"[RAG Pipeline Warning] Could not initialize Gemini client: {e}")

    def call_llm(self, prompt: str, retries: int = 3) -> str:
        """Call Gemini API with dynamic RAG prompt and retry logic."""
        if not self.client:
            return ""
        
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=config.MODEL_NAME,
                    contents=prompt,
                    config={
                        'system_instruction': SYSTEM_PROMPT_RAG,
                        'temperature': 0.1,
                        'response_mime_type': 'application/json'
                    }
                )
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_sec = 5 * (attempt + 1)
                    print(f"[API Rate Limit 429] Waiting {wait_sec}s before retry {attempt + 1}/{retries}...")
                    time.sleep(wait_sec)
                else:
                    print(f"[RAG LLM Call Error] {e}")
                    break
        return ""

    def process_case_rag(self, query_case: Dict[str, str], retrieved_examples: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Process query_case using retrieved dynamic few-shot examples.
        """
        template_content = query_case.get('template_content', '')

        if self.client and retrieved_examples:
            prompt = build_rag_prompt(query_case, retrieved_examples)
            raw_response = self.call_llm(prompt)
            if raw_response:
                try:
                    data = json.loads(raw_response)
                    field_updates = data.get('field_updates', {})
                    impression = data.get('impression', '')
                    report = TemplateEditor.render_report(template_content, field_updates, impression)
                    return report, data
                except Exception as e:
                    print(f"[RAG JSON Parse Error] {e}")

        # Fallback to heuristic pipeline if API key is not configured or API fails
        return self.fallback_pipeline.process_case(query_case)
