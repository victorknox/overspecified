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
    get_contrasting_detection_system_message,
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


class AttributeValue(BaseModel):
    value: str = Field(..., description="The attribute value")
    

class Attribute(BaseModel):
    name: str = Field(..., description="Attribute name")
    values: List[str] = Field(default_factory=list, description="List of possible values")
    is_contrasting: bool = Field(..., description="True if values are ordinal/sortable, False if diverse")


class Token(BaseModel):
    text: str = Field(..., description="The original token text")
    is_modifiable: bool = Field(..., description="True if this token is an object that can be modified with attributes")
    attributes: List[Attribute] = Field(default_factory=list, description="List of attributes for this token")


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

    def detect_attribute_contrasting_potential(
        self,
        obj: str,
        attribute: str
    ) -> bool:
        system_message = get_contrasting_detection_system_message()
        user_message = f"Object: {obj}\nAttribute: {attribute}\n\nDoes this attribute have contrasting/ordinal potential?"
        
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
            return False
            
        try:
            result = json.loads(content.strip())
            return result.get("is_contrasting", False)
        except Exception as e:
            print(f"Error parsing contrasting potential detection: {e}")
            return False

    def expand_attribute_values(
        self,
        word: str,
        attribute: str,
        is_contrasting: Optional[bool] = None
    ) -> List[str]:
        cache_key = (word, attribute, self.max_values_per_attribute, is_contrasting if is_contrasting is not None else self.contrasting)
        if cache_key in self._attribute_value_cache:
            return self._attribute_value_cache[cache_key]
        
        use_contrasting = is_contrasting if is_contrasting is not None else self.contrasting
        system_content = get_attribute_value_expansion_system_message(
            self.max_values_per_attribute, 
            use_contrasting
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

    def generate_token_structure(self, verbose: bool = False) -> List[Token]:
        """
        Generate the token-based structure where each word in the prompt becomes a Token
        with is_modifiable flag and attributes if applicable.
        
        Pipeline:
        1. Tokenize input by whitespace
        2. Extract attributes from prompt
        3. For each attribute, detect if it CAN be contrasting (before value generation)
        4. Expand attribute values using the contrasting information
        5. Build Token structure
        """
        if verbose:
            print(f"Original prompt: {self.prompt}\n")
        
        # Step 1: Tokenize the input by whitespace
        words = self.prompt.split()
        
        # Step 2: Extract attributes from the prompt
        extracted_attributes = self.extract_attributes()
        if not extracted_attributes:
            print("Failed to extract attributes.")
            return [Token(text=word, is_modifiable=False, attributes=[]) for word in words]
        
        if verbose:
            print("=" * 60)
            print("STEP 1: Extracted attributes from prompt")
            print("=" * 60)
            for obj_dict in extracted_attributes:
                print(f"  {obj_dict}")
            print()
        
        # Step 3: Build a mapping of objects to their attributes AND detect contrasting potential
        object_attr_map: Dict[str, List[str]] = {}
        object_contrasting_map: Dict[str, Dict[str, bool]] = {}
        
        for obj_dict in extracted_attributes:
            for obj, attrs in obj_dict.items():
                object_attr_map[obj.lower()] = attrs
                object_contrasting_map[obj.lower()] = {}
                
                if verbose:
                    print("=" * 60)
                    print(f"STEP 2: Detecting contrasting potential for '{obj}'")
                    print("=" * 60)
                
                for attr_name in attrs:
                    is_contrasting = self.detect_attribute_contrasting_potential(obj, attr_name)
                    object_contrasting_map[obj.lower()][attr_name] = is_contrasting
                    
                    if verbose:
                        print(f"  {attr_name}: contrasting={is_contrasting}")
                
                if verbose:
                    print()
        
        # Step 4 & 5: Create Token objects for each word
        if verbose:
            print("=" * 60)
            print("STEP 3: Expanding attribute values and building tokens")
            print("=" * 60)
        
        tokens = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            
            if word_lower in object_attr_map:
                if verbose:
                    print(f"\nProcessing modifiable token: '{word}'")
                
                attribute_list = []
                for attr_name in object_attr_map[word_lower]:
                    is_contrasting = object_contrasting_map[word_lower].get(attr_name, False)
                    values = self.expand_attribute_values(word_lower, attr_name, is_contrasting)
                    
                    if values:
                        attribute_list.append(Attribute(
                            name=attr_name,
                            values=values,
                            is_contrasting=is_contrasting
                        ))
                        
                        if verbose:
                            print(f"  {attr_name} (contrasting={is_contrasting}): {values}")
                
                tokens.append(Token(
                    text=word,
                    is_modifiable=True,
                    attributes=attribute_list
                ))
            else:
                # other non-modifiable tokens
                tokens.append(Token(
                    text=word,
                    is_modifiable=False,
                    attributes=[]
                ))
        
        if verbose:
            print()
        
        return tokens

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
