# Few-shot examples for attribute extraction
ATTRIBUTE_EXTRACTION_EXAMPLES = [
    {
        "prompt": "Draw a young man riding a blue bicycle in a big park.",
        "output": {
            "items": [
                {"object": "man", "attributes": ["age", "ethnicity", "origin", "athleticism", "profession", "emotion", "clothing"]},
                {"object": "riding", "attributes": ["speed", "style", "intensity"]},
                {"object": "bicycle", "attributes": ["color", "type", "brand", "condition"]},
                {"object": "park", "attributes": ["size", "location", "features"]}
            ]
        }
    },
    {
        "prompt": "Generate a photo of a white dog playing in a snowy field.",
        "output": {
            "items": [
                {"object": "dog", "attributes": ["breed", "color", "age", "size", "emotion", "activity"]},
                {"object": "playing", "attributes": ["energy", "intensity", "style"]},
                {"object": "field", "attributes": ["coverage", "weather", "size"]}
            ]
        }
    }
]

# Few-shot examples for attribute value expansion
ATTRIBUTE_VALUE_EXAMPLES = [
    {
        "object": "person",
        "attribute": "age",
        "expansions": ["infant", "child", "teenager", "adult", "elderly"]
    },
    {
        "object": "person",
        "attribute": "emotion",
        "expansions": ["happy", "sad", "angry", "surprised", "neutral"]
    },
    {
        "object": "walking",
        "attribute": "speed",
        "expansions": ["slow", "brisk", "hurried", "leisurely", "normal"]
    }
]

PROMPT_EXPANSION_EXAMPLES = [
    {
        "original_prompt": "Draw a picture of a person walking.",
        "attribute_map": {"person": {"age": "young"}},
        "rewritten_prompt": "Draw a picture of a young person walking."
    },
    {
        "original_prompt": "Draw a picture of a car.",
        "attribute_map": {"car": {"color": "red"}},
        "rewritten_prompt": "Draw a picture of a red car."
    },
    {
        "original_prompt": "Create an image of a tree.",
        "attribute_map": {"tree": {"season": "autumn"}},
        "rewritten_prompt": "Create an image of an autumn tree."
    },
    {
        "original_prompt": "A person cooking.",
        "attribute_map": {"person": {"age": "elderly", "gender": "female"}},
        "rewritten_prompt": "An elderly female person cooking."
    }
]


def get_attribute_extraction_system_message(max_attributes: int) -> str:
    """Generate system message for attribute extraction."""
    return (
        f"You are an expert word and property extractor for visual image generation. For the following prompt, extract all modifiable words—these can be nouns, verbs, adjectives, adverbs, or any other word that could reasonably have visually depictable properties. "
        f"For each modifiable word you identify, map it to a general set of possible attribute THEMES, based on common visual classes for that kind of word. "
        f"IMPORTANT: Only include as attribute themes those properties that have a clear, visualizable impact on an image (for example: color, size, lighting, material, facial expression, pose, body type, pattern, clothing, style, setting, etc.). IGNORE non-visual, abstract, or temporal properties such as 'duration', 'popularity', 'history', or anything that cannot be shown visually in a single image. "
        f"For example, for the noun 'man', attribute themes may include: age, ethnicity, origin, athleticism, profession, clothing, emotion. For the verb 'walk', only visual properties such as: speed (if it affects pose), style (if visually distinct), intensity (if visible), etc. Adverbs or other words should also be included ONLY if they can be shown visually, such as 'quickly' suggesting pose or motion blur. "
        f"For each modifiable word you detect, list as attributes at most {max_attributes} broad, visually-representable attribute themes relevant to that word, not specific values from the prompt. "
        f"Return ONLY a JSON object that passes the following Pydantic schema:\n\n"
        "class ObjectAttributes(BaseModel):\n"
        "    object: str  # the modifiable word (noun, verb, adjective, adverb, etc.)\n"
        "    attributes: List[str]  # general THEMES of visually depictable properties for this word\n"
        "class ExtractionResult(BaseModel):\n"
        "    items: List[ObjectAttributes]\n"
        "\n"
        "EXAMPLES (few-shot):\n\n"
        f"Prompt: '{ATTRIBUTE_EXTRACTION_EXAMPLES[0]['prompt']}'\n"
        f"{_format_json(ATTRIBUTE_EXTRACTION_EXAMPLES[0]['output'])}\n\n"
        f"Prompt: '{ATTRIBUTE_EXTRACTION_EXAMPLES[1]['prompt']}'\n"
        f"{_format_json(ATTRIBUTE_EXTRACTION_EXAMPLES[1]['output'])}\n\n"
        f"For each word, DO NOT include more than {max_attributes} attribute themes. "
        "Only return a flat list of all detected modifiable words and a list of their visually depictable attribute THEMES. "
        "If a word appears repeatedly, merge all relevant visual themes, but never list more than the max per word. "
        "Exclude non-visual attribute themes such as 'duration', 'frequency', 'popularity', etc. "
        "Do not include any explanations, extra text, parse trees, or commentary—just a JSON object as specified."
    )


def get_attribute_value_expansion_system_message(max_values: int, contrasting: bool = False) -> str:
    """Generate system message for attribute value expansion."""
    contrast_instruction = ""
    if contrasting:
        contrast_instruction = (
            f"Generate {max_values} HIGHLY CONTRASTING and DIVERSE values that represent opposite or very different extremes of the attribute spectrum. "
            'For example, for object: "person", attribute: "age", you might return: '
            '{"object": "person", "attribute": "age", "expansions": ["baby", "teenager", "adult", "elderly"]}'
        )
    else:
        contrast_instruction = f"Generate {max_values} diverse, representative values covering the range of this attribute. "
    return (
        "You are an expert at extracting visually depictable attribute VALUEs for attributes of objects, "
        "in a way that would assist an artist depicting a given object. "
        f"Given an object and one of its attribute themes, return a JSON array of values. "
        f"{contrast_instruction}"
        "Output format: {\"object\": <object>, \"attribute\": <attribute>, \"expansions\": [values]}.\n\n"
        "Only include concrete, visually depictable values. Never include explanations, just output the JSON."
    )


def get_prompt_expansion_system_message() -> str:
    """Generate system message for prompt expansion."""
    return (
        "You are a careful prompt rewriter for image generation. "
        "Add the specified attribute values to the original prompt in a natural, minimal way. "
        "Keep the original sentence structure intact. Add only the given attributes as simple modifiers. "
        "Do NOT add extra details, explanations, or change the core meaning. Keep it concise."
    )


def get_contrasting_detection_system_message() -> str:
    """Generate system message for detecting if an attribute has contrasting/ordinal potential."""
    return (
        "You are an expert at analyzing attributes to determine if they have ordinal/contrasting potential.\n\n"
        "An attribute has CONTRASTING POTENTIAL if it represents a spectrum or scale that can go from one extreme to another:\n"
        "- Examples: age (young to old), size (small to large), brightness (dark to light), temperature (cold to hot)\n\n"
        "An attribute is DIVERSE (not contrasting) if it represents categorical variety with no inherent order:\n"
        "- Examples: ethnicity, country, pattern, flavor, type, style\n\n"
        'Respond with a JSON object: {"is_contrasting": true/false, "reasoning": "brief explanation"}'
    )


def _format_json(data: dict) -> str:
    """Format dictionary as JSON string for prompts."""
    import json
    return json.dumps(data, indent=2)

