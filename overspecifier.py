from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import openai
import json
import random
import os
from prompts import (
    get_attribute_extraction_system_message,
    get_attribute_value_expansion_system_message,
    get_prompt_expansion_system_message,
    ATTRIBUTE_VALUE_EXAMPLES,
    PROMPT_EXPANSION_EXAMPLES
)

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ObjectAttributes(BaseModel):
    object: str = Field(..., description="Primary object extracted")
    attributes: List[str] = Field(..., description="List of attributable words or phrases")


class ExtractionResult(BaseModel):
    items: List[ObjectAttributes]


class Overspecifier:
    def __init__(
        self,
        prompt: str,
        model: str = "gpt-4.1-nano",
        max_attrs_per_object: int = 2,
        max_values_per_attribute: int = 5,
        contrasting: bool = False,
        max_prompt_expansion: int = 5,
        seed: Optional[int] = None
    ):
        self.prompt = prompt
        self.model = model
        self.max_attrs_per_object = max_attrs_per_object
        self.max_values_per_attribute = max_values_per_attribute
        self.contrasting = contrasting
        self.max_prompt_expansion = max_prompt_expansion
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self._attribute_value_cache: Dict[tuple, List[str]] = {}

    def extract_attributes(self, prompt: Optional[str] = None) -> Optional[List[Dict[str, List[str]]]]:
        if prompt is None:
            prompt = self.prompt
            
        system_message = get_attribute_extraction_system_message(7)
        user_message = f"Prompt: {prompt}\n\nExtract the objects and their attribute THEMES according to the instructions and schema."
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            seed=self.seed
        )
        
        content = response.choices[0].message.content
        if content is None:
            return None
            
        result_json = content.strip()
        try:
            data = json.loads(result_json)
            out = [{item['object']: item['attributes']} for item in data.get('items', [])]
            return out
        except Exception as e:
            print(f"Error parsing: {e}")
            return None

    def expand_attribute_values(
        self,
        word: str,
        attribute: str
    ) -> List[str]:
        cache_key = (word, attribute, self.max_values_per_attribute, self.contrasting)
        if cache_key in self._attribute_value_cache:
            return self._attribute_value_cache[cache_key]
        
        system_content = get_attribute_value_expansion_system_message(
            self.max_values_per_attribute, 
            self.contrasting
        )
        messages = [{"role": "system", "content": system_content}]
        
        for ex in ATTRIBUTE_VALUE_EXAMPLES:
            user_msg = f"Object: {ex['object']}\nAttribute: {ex['attribute']}\nList visually distinct values."
            assistant_msg = json.dumps({
                'object': ex['object'],
                'attribute': ex['attribute'],
                'expansions': ex['expansions']
            })
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        current_user_msg = f"Object: {word}\nAttribute: {attribute}\nList visually distinct values."
        messages.append({"role": "user", "content": current_user_msg})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=self.seed
        )
        
        content = response.choices[0].message.content
        if content is None:
            return []
            
        result = content.strip()
        try:
            expanded = json.loads(result.replace("'", '"'))
            values = expanded.get("expansions", [])[:self.max_values_per_attribute]
            self._attribute_value_cache[cache_key] = values
            return values
        except Exception as e:
            print(f"Error parsing expansion: {e}")
            return []

    def expand_prompt(
        self,
        prompt: Optional[str] = None,
        attribute_map: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        if prompt is None:
            prompt = self.prompt
        if attribute_map is None:
            attribute_map = {}
            
        example_messages = []
        for ex in PROMPT_EXPANSION_EXAMPLES:
            user_txt = f"Original: {ex['original_prompt']}\nAttributes: {json.dumps(ex['attribute_map'])}"
            assistant_txt = ex["rewritten_prompt"]
            example_messages.append({"role": "user", "content": user_txt})
            example_messages.append({"role": "assistant", "content": assistant_txt})
        
        user_message = f"Original: {prompt}\nAttributes: {json.dumps(attribute_map)}"
        
        system_message = {
            "role": "system",
            "content": get_prompt_expansion_system_message()
        }
        
        messages = [system_message] + example_messages + [{"role": "user", "content": user_message}]
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=self.seed
        )
        
        content = response.choices[0].message.content
        if content is None:
            return prompt
            
        new_prompt = content.strip()
        return new_prompt

    def generate_random_attribute_map(
        self,
        extracted_attributes: List[Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, str]]:
        attribute_map = {}
        
        for obj_dict in extracted_attributes:
            for obj, attrs in obj_dict.items():
                attribute_map[obj] = {}
                
                selected_attrs = random.sample(attrs, min(self.max_attrs_per_object, len(attrs)))
                
                for attr in selected_attrs:
                    values = self.expand_attribute_values(obj, attr)
                    
                    if values:
                        selected_value = random.choice(values)
                        attribute_map[obj][attr] = selected_value
        
        return attribute_map

    def pipeline(self, verbose: bool = False) -> List[str]:
        if verbose:
            print(f"Original prompt: {self.prompt}\n")
        
        extracted_attributes = self.extract_attributes()
        if not extracted_attributes:
            print("Failed to extract attributes.")
            return []
        
        if verbose:
            print("Extracted attributes:")
            for obj_dict in extracted_attributes:
                print(f"  {obj_dict}")
            print()
        
        if verbose:
            print("Expanding attribute values...")
        
        for obj_dict in extracted_attributes:
            for obj, attrs in obj_dict.items():
                for attr in attrs:
                    values = self.expand_attribute_values(obj, attr)
                    if verbose:
                        print(f"  {obj}.{attr}: {values}")
        
        if verbose:
            print()
        
        expanded_prompts = []
        for i in range(self.max_prompt_expansion):
            attribute_map = self.generate_random_attribute_map(extracted_attributes)
            
            if verbose:
                print(f"Variation {i+1} attribute map:")
                print(f"  {json.dumps(attribute_map, indent=2)}")
            
            expanded = self.expand_prompt(attribute_map=attribute_map)
            expanded_prompts.append(expanded)
            
            if verbose:
                print(f"  -> {expanded}\n")
        
        return expanded_prompts
