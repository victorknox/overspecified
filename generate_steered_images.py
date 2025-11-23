import sys
import os
import subprocess
import random
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

# Add attribute-control to path
# Assuming this script is in overspecified/ and attribute-control is in overspecified/attribute-control
sys.path.append(os.path.join(os.path.dirname(__file__), 'attribute-control'))

try:
    from attribute_control import EmbeddingDelta
    from attribute_control.model import SDXL
    from attribute_control.prompt_utils import get_mask_regex
except ImportError as e:
    print(f"Error: Could not import attribute_control. {e}")
    # print("Error: Could not import attribute_control. Please ensure the 'attribute-control' repository is cloned in the same directory.")
    sys.exit(1)

# --- Configuration ---
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE != 'cpu' else torch.float32
PRETRAINED_DELTAS_DIR = os.path.join(os.path.dirname(__file__), 'attribute-control', 'pretrained_deltas')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_steered_images')

# --- Dataset ---
DATASET = [
    'a person', 'a bed', 'a bike', 'a car', 'a chair', 'a table', 'a truck', 'A person is asleep in a bed.', 'A person rides a bike.', 'A person is driving a car.', 
    'A person is sitting on a chair.', 'A person is eating at a table.', 'A person unloads a truck.', 
    'A bed and a chair are in the same room.', 'A small table stands beside a bed.', 
    'A bed is interacting with a bike.', 'A bed is interacting with a car.', 'A bed is interacting with a truck.', 
    'A bike is strapped to the roof of a car.', 'A bike is parked near a chair.', 'A bike is parked near a table.', 
    'A bike is interacting with a truck.', 'A car is parked near a chair.', 'A car is parked near a table.', 
    'A car follows a truck on the highway.', 'A chair is placed against a bed.', 'A chair is interacting with a truck.', 
    'A table is placed against a bed.', 'A person sits on a chair beside a table.', 'A person is working on a table while sitting in bed.', 
    'A person is used to move a bike near a car.', 'A person is used to move a bike near a chair.', 
    'A person is used to move a bike near a table.', 'A person is used to move a bike near a truck.', 
    'A person is used to move a car near a chair.', 'A person is used to move a car near a table.', 
    'A person exits a car after being towed by a truck.', 'A person is used to move a chair near a table.', 
    'A person is used to move a chair near a truck.', 'A person is used to move a table near a truck.', 
    'A bed is used to move a bike near a car.', 'A bed is used to move a bike near a chair.', 
    'A bed is used to move a bike near a table.', 'A bed is used to move a bike near a truck.', 
    'A bed is used to move a car near a chair.', 'A bed is used to move a car near a table.', 
    'A bed is used to move a car near a truck.', 'A bed is used to move a chair near a table.', 
    'A bed is used to move a chair near a truck.', 'A bed is used to move a table near a truck.', 
    'A bike is strapped to the roof of a car.', 'A bike is stored in a garage containing a chair and a table.', 
    'A bike is used to move a chair near a truck.', 'A bike is used to move a table near a truck.', 
    'A bike is interacting with a car.', 'A car is stored in a garage containing a bed and a chair.', 
    'A car is stored in a garage containing a bed and a table.', 'A car follows a truck on the highway.', 
    'A car is used to move a chair near a truck.', 'A car is used to move a table near a truck.', 
    'A chair is stored in a garage containing a bed and a table.', 'A chair is used to move a table near a truck.', 
    'A table is used to move a truck near a bed.'
]

# --- Attributes Dictionary ---
# Maps entity name to list of attribute filenames (without extension)
ATTRIBUTES = {
    'bed': ['bed_age', 'bed_price', 'bed_size'],
    'bike': ['bike_age', 'bike_price', 'bike_size'],
    'car': ['car_age', 'car_price'],
    'chair': ['chair_age', 'chair_price'],
    'person': [
        'person_age', 'person_bald', 'person_colorful_outfit', 'person_curly_hair', 
        'person_elegant', 'person_fitness', 'person_freckled', 'person_groomed', 
        'person_height', 'person_long_hair', 'person_makeup', 'person_pale', 
        'person_pierced', 'person_posture', 'person_scarred', 'person_smile', 
        'person_surprised', 'person_tattooed', 'person_tired', 'person_width'
    ],
    'table': ['table_age', 'table_price', 'table_size'],
    'truck': ['truck_price']
}

def setup_model():
    print(f"Loading SDXL model on {DEVICE}...")
    model = SDXL(
        pipeline_type='diffusers.StableDiffusionXLPipeline',
        model_name='stabilityai/stable-diffusion-xl-base-1.0',
        pipe_kwargs={'torch_dtype': DTYPE, 'variant': 'fp16', 'use_safetensors': True},
        device=DEVICE
    )
    return model

def get_valid_steering_options(prompt):
    """
    Finds which entities from our ATTRIBUTES dict are present in the prompt.
    Returns a list of (entity, attribute_name, attribute_file) tuples.
    """
    options = []
    prompt_lower = prompt.lower()
    for entity, attrs in ATTRIBUTES.items():
        # Simple check if entity name is in prompt. 
        # Note: This might match 'car' in 'scar', but for this dataset it's likely fine.
        # Using regex word boundary for better accuracy.
        mask = get_mask_regex(prompt_lower, f'\\b({entity})\\b')
        if mask is not None and np.any(mask):
             for attr in attrs:
                 options.append((entity, attr))
    return options

def generate_image(model, prompt, entity, attribute_name, scale, seed, output_path):
    delta_path = os.path.join(PRETRAINED_DELTAS_DIR, f"{attribute_name}.pt")
    if not os.path.exists(delta_path):
        print(f"Warning: Delta file not found at {delta_path}")
        return

    # Load Delta
    delta = EmbeddingDelta(model.dims)
    state_dict = torch.load(delta_path, map_location='cpu') # Load to CPU first
    delta.load_state_dict(state_dict['delta'])
    delta = delta.to(DEVICE)

    # Prepare steering
    pattern_target = f'\\b({entity})\\b'
    # Use case-insensitive matching if needed, but regex usually is case sensitive unless specified.
    # The dataset has "A person...", so "person" matches.
    # But "A car..." -> "car" matches.
    # Let's try using prompt as is.
    characterwise_mask = get_mask_regex(prompt, pattern_target)
    
    if characterwise_mask is None or not np.any(characterwise_mask):
        # Try lowercase prompt if original failed (though get_mask_regex might handle it?)
        # But we need to pass the original prompt to the model.
        # If mask generation fails on original prompt, maybe we should try to adjust pattern?
        # For now, just warn.
        print(f"Warning: Could not find entity '{entity}' in prompt '{prompt}' for masking (mask empty).")
        return

    emb = model.embed_prompt(prompt)
    emb_neg = None # Optional: Add negative prompt if needed

    # Generate
    # Using delay_relative=0.2 as per notebook recommendation for minor changes
    delay_relative = 0.2 
    
    with torch.no_grad():
        img = model.sample_delayed(
            embs=[delta.apply(emb, characterwise_mask, scale)],
            embs_unmodified=[emb],
            embs_neg=[emb_neg],
            delay_relative=delay_relative,
            generator=torch.manual_seed(seed),
            guidance_scale=7.5
        )[0]

    img.save(output_path)
    # print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate steered images.")
    parser.add_argument('--test', action='store_true', help="Run a quick test with 1 image.")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = setup_model()
    
    dataset_to_use = DATASET
    if args.test:
        dataset_to_use = DATASET[:1]
        print("Running in TEST mode (1 prompt only)")

    for i, prompt in enumerate(tqdm(dataset_to_use, desc="Processing Prompts")):
        # Create a subfolder for each prompt to keep things organized? 
        # Or just filename convention: {prompt_id}_{variation_id}_{entity}_{attribute}_{scale}.png
        
        # Clean prompt for filename
        safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
        
        valid_options = get_valid_steering_options(prompt)
        
        if not valid_options:
            print(f"Skipping prompt '{prompt}': No steerable entities found.")
            continue

        num_variations = 1 if args.test else 10
        
        for v in range(num_variations):
            # Randomly sample steering parameters
            entity, attribute_name = random.choice(valid_options)
            scale = random.uniform(-2.0, 2.0)
            seed = random.randint(0, 1000000)
            
            filename = f"{i:03d}_var{v:02d}_{entity}_{attribute_name}_s{scale:.2f}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            print(f"Generating: {prompt} | Steer: {entity} -> {attribute_name} (scale {scale:.2f})")
            
            try:
                generate_image(model, prompt, entity, attribute_name, scale, seed, output_path)
            except Exception as e:
                print(f"Failed to generate for '{prompt}': {e}")

    print(f"Done! Images saved to {OUTPUT_DIR}")

    print("Stopping RunPod instance...")
    subprocess.run(["runpodctl", "stop", "6p80cr5t6999y2"])

if __name__ == "__main__":
    main()
