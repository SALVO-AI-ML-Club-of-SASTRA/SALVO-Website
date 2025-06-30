
import concurrent.futures
# Create your views here.
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os, random, json
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
# Create your views here.



DEFAULT_MODEL_FILENAME= "step_176000.keras"
# === Global Constants ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_FILENAME)
MODELS_DIR = os.path.join(settings.MEDIA_ROOT, 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Global dictionary to store loaded models
loaded_models = {}
model_lock = threading.Lock()

# Load default model only once
try:
    default_model = load_model(MODEL_PATH)
    loaded_models['default'] = {
        'model': default_model,
        'name': 'Default Model',
        'filename': DEFAULT_MODEL_FILENAME
    }
except Exception as e:
    print(f"Error loading default model: {e}")
    default_model = None

BASE_DIR = os.path.dirname(__file__)
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")

with open(CLASSES_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

def load_model_safe(model_path):
    """Safely load a model with error handling"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def get_available_models():
    """Get list of available model files"""
    models = []
    
    # Add default model if it exists
    if 'default' in loaded_models:
        models.append({
            'filename': DEFAULT_MODEL_FILENAME,
            'name': 'Default Model',
            'uploaded_at': datetime.now().isoformat(),
            'size': None,
            'is_default': True
        })
    
    # Add uploaded models
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.keras'):
                if filename == DEFAULT_MODEL_FILENAME:
                    continue  # Skip listing default model again
                filepath = os.path.join(MODELS_DIR, filename)
                try:
                    stat = os.stat(filepath)
                    models.append({
                        'filename': filename,
                        'name': filename.replace('.keras', '').replace('_', ' ').title(),
                        'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'size': stat.st_size,
                        'is_default': False
                    })
                except Exception as e:
                    print(f"Error getting stats for {filename}: {e}")
    
    return models

def load_all_models():
    """Load all available models into memory"""
    global loaded_models
    
    with model_lock:
        # Keep default model
        if 'default' not in loaded_models and default_model:
            loaded_models['default'] = {
                'model': default_model,
                'name': 'Default Model',
                'filename': DEFAULT_MODEL_FILENAME
            }
        
        # Load uploaded models
        if os.path.exists(MODELS_DIR):
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith('.keras'):
                    if filename == DEFAULT_MODEL_FILENAME:
                        continue  # Skip loading default model again
                    model_key = filename.replace('.keras', '')
                    if model_key not in loaded_models:
                        filepath = os.path.join(MODELS_DIR, filename)
                        model = load_model_safe(filepath)
                        if model:
                            loaded_models[model_key] = {
                                'model': model,
                                'name': filename.replace('.keras', '').replace('_', ' ').title(),
                                'filename': filename
                            }
                            print(f"Loaded model: {filename}")

def image_to_base64(img):
    """Converts a single-channel image array to base64 PNG."""
    pil_img = Image.fromarray((img * 255).astype(np.uint8)).convert("L")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def username_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        request.session['username'] = username
        return redirect('models')  # Changed to redirect to model management
    return render(request, 'drawapp/username_page.html')

def model_management(request):
    """Model management page"""
    if 'username' not in request.session:
        return redirect('username')
    
    return render(request, 'drawapp/model_management.html')

@csrf_exempt
def upload_model(request):
    """Handle model file upload"""
    print("📥 Received model upload request.")
    
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST request required'})
    
    if 'model_file' not in request.FILES:
        print("❌ No file received in upload.")
        return JsonResponse({'success': False, 'error': 'No file uploaded'})
    
    uploaded_file = request.FILES['model_file']
    
    if not uploaded_file.name.endswith('.keras'):
        print(f"❌ Invalid file type: {uploaded_file.name}")
        return JsonResponse({'success': False, 'error': 'File must be a .keras file'})
    
    try:
        filename = uploaded_file.name
        if filename == DEFAULT_MODEL_FILENAME:
            return JsonResponse({'success': False, 'error': 'Filename conflicts with default model'})

        filepath = os.path.join(MODELS_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"⚠️ Model already exists: {filename}")
            return JsonResponse({'success': False, 'error': 'Model with this name already exists'})
        
        print(f"📂 Saving uploaded model to: {filepath}")
        with open(filepath, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        print(f"🔍 Attempting to load uploaded model for validation: {filepath}")
        test_model = load_model_safe(filepath)
        if test_model is None:
            print("❌ Uploaded file is not a valid Keras model. Deleting.")
            os.remove(filepath)
            return JsonResponse({'success': False, 'error': 'Invalid .keras file or incompatible model'})
        
        model_key = filename.replace('.keras', '')
        with model_lock:
            loaded_models[model_key] = {
                'model': test_model,
                'name': filename.replace('.keras', '').replace('_', ' ').title(),
                'filename': filename
            }

        print(f"✅ Model '{filename}' uploaded and loaded successfully.")
        return JsonResponse({'success': True})
        
    except Exception as e:
        print(f"❌ Exception during model upload: {e}")
        return JsonResponse({'success': False, 'error': f'Upload failed: {str(e)}'})


@csrf_exempt
def get_models(request):
    """Get list of available models"""
    try:
        models = get_available_models()
        print("Current models in directory:", models)
        return JsonResponse({'models': models})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def delete_model(request):
    """Delete a model file"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST request required'})
    
    try:
        data = json.loads(request.body)
        filename = data.get('filename')
        
        if not filename:
            return JsonResponse({'success': False, 'error': 'Filename required'})
        
        # Don't allow deleting default model
        if filename == 'step_176000.keras':
            return JsonResponse({'success': False, 'error': 'Cannot delete default model'})
        
        filepath = os.path.join(MODELS_DIR, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            
            # Remove from loaded models
            model_key = filename.replace('.keras', '')
            with model_lock:
                if model_key in loaded_models:
                    del loaded_models[model_key]
            
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': 'File not found'})
    
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    


# === Global Constants ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "step_176000.keras")

# Load model only once
from threading import Lock
models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')])


# Thread-safe model loading
model_lock = Lock()
models = []
with model_lock:
    for f in model_files:
        try:
            models.append(load_model(os.path.join(models_dir, f)))
            print(f"✅ Loaded model: {f}")
        except Exception as e:
            print(f"❌ Failed to load model {f}: {e}")

BASE_DIR = os.path.dirname(__file__)
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")

with open(CLASSES_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
    
def image_to_base64(img):
    """Converts a single-channel image array to base64 PNG."""
    pil_img = Image.fromarray((img * 255).astype(np.uint8)).convert("L")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")




def draw_page(request):
    if 'username' not in request.session:
        return redirect('username')

    # Select 3 random classes for this session
    challenge_words = random.sample(CLASSES, 3)
    request.session['challenge_words'] = challenge_words

    # Pass readable names to the template
    readable_words = [w.replace('_', ' ').title() for w in challenge_words]
    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')])
    model_mapping = {i + 1: model_files[i] for i in range(len(model_files))}
    return render(request, 'drawapp/draw_page.html', {
        'challenge_words': readable_words,"model_mapping": model_mapping
    })
    

import random

def extract_and_resize_parts(base64_img, size=(128, 128), rotation_prob=0.3):
    header, encoded = base64_img.split(",", 1)
    pil_img = Image.open(BytesIO(base64.b64decode(encoded))).convert('L')
    full_np = np.array(pil_img).astype(np.float32)

    # 1. Original resized
    original = cv2.resize(full_np, size)

    # 2. Three overlapping parts
    h, w = full_np.shape
    thirds = []
    step = int(h / 4)
    for i in range(3):
        crop = full_np[i*step:i*step+int(h/2), i*step:i*step+int(w/2)]
        crop = cv2.resize(crop, size)
        thirds.append(crop)

    # 3. Center crop
    ch, cw = 390, 390
    top = (h - ch) // 2
    left = (w - cw) // 2
    center_crop = full_np[top:top+ch, left:left+cw]
    center_resized = cv2.resize(center_crop, size)

    # 4. Optional 90-degree rotation on central 400x400 part
    rotation_aug = None
    if random.random() < rotation_prob:
        rot_ch, rot_cw = 400, 400
        rt = (h - rot_ch) // 2
        rl = (w - rot_cw) // 2
        rot_crop = full_np[rt:rt+rot_ch, rl:rl+rot_cw]
        rotated = cv2.rotate(rot_crop, cv2.ROTATE_90_CLOCKWISE)
        rotation_aug = cv2.resize(rotated, size)

    # Stack all for prediction
    all_images = [original] + thirds + [center_resized]
    if rotation_aug is not None:
        all_images.append(rotation_aug)

    all_images = [np.expand_dims(img, axis=(0, -1)) / 255.0 for img in all_images]  # shape: (1, 128, 128, 1)

    return all_images # list of 5 image tensors + optional rotation augmentation


def preprocess_base64_image(image_data):
    header, encoded = image_data.split(",", 1)
    image = Image.open(BytesIO(base64.b64decode(encoded))).convert('L')
    image = image.resize((128, 128))  # match model input
    img_arr = np.array(image).astype(np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=(0, -1))  # shape: (1, 128, 128, 1)
    return img_arr



def filter_predictions_by_hints(candidates, target_word, hints):
    """
    Filter predictions based on hints with standardized normalization.
    
    Args:
        candidates: List of candidate words from model predictions
        target_word: The actual word being drawn (from frontend)
        hints: List of hint dictionaries with 'index' and 'letter' keys
    
    Returns:
        List of filtered candidate words
    """
    def normalize_word(word):
        """Standardized normalization: remove underscores and spaces, convert to lowercase"""
        return word.replace('_', '').replace(' ', '').lower()
    
    # Normalize target word
    target_normalized = normalize_word(target_word)
    print(f"\n\nTarget word: '{target_word}' -> normalized: '{target_normalized}'")
    print(f"Hints received: {hints}")
    
    # Step 1: Filter by length first
    length_filtered = []
    for candidate in candidates:
        candidate_normalized = normalize_word(candidate)
        if len(candidate_normalized) == len(target_normalized):
            length_filtered.append(candidate)
    
    print(f"After length filter ({len(target_normalized)} chars): {[normalize_word(w) for w in length_filtered]}")
    
    # Step 2: Apply hint filters
    hint_filtered = length_filtered.copy()
    
    for hint in hints:
        hint_index = hint.get('index')
        hint_letter = hint.get('letter', '').lower()
        
        print(f"Applying hint: index={hint_index}, letter='{hint_letter}'")
        
        # Validate hint index
        if hint_index is None or hint_index < 0 or hint_index >= len(target_normalized):
            print(f"⚠️ Invalid hint index {hint_index} for word length {len(target_normalized)}")
            continue
        
        # Filter candidates that match this hint
        temp_filtered = []
        for candidate in hint_filtered:
            candidate_normalized = normalize_word(candidate)
            
            # Check if candidate has the correct letter at the hint position
            if (len(candidate_normalized) > hint_index and 
                candidate_normalized[hint_index] == hint_letter):
                temp_filtered.append(candidate)
        
        hint_filtered = temp_filtered
        print(f"After applying hint {hint}: {[normalize_word(w) for w in hint_filtered]}")
    
    print(f"Final filtered results: {[normalize_word(w) for w in hint_filtered]}\n\n")
    return hint_filtered


def filter_by_length_only(candidates, target_word):
    """Filter candidates by length only (fallback function)"""
    def normalize_word(word):
        return word.replace('_', '').replace(' ', '').lower()
    
    target_length = len(normalize_word(target_word))
    return [c for c in candidates if len(normalize_word(c)) == target_length]


def is_blank(img_array, threshold=0.98):
    # If RGB, flatten to grayscale
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=-1)

    # Check ratio of near-white pixels (>0.98 for normalized)
    white_pixel_ratio = np.sum(img_array > threshold) / img_array.size
    #print(f"🔍 White pixel ratio: {white_pixel_ratio:.2f}")

    return white_pixel_ratio > 0.98


RANDOM_GUESSES = [
                "Toaster", "Flying cat", "Unicorn spaghetti", "Invisible car",
                "A mistake?", "404 drawing not found", "Quantum blob", "White noise", "Just... nothing"
            ]

@csrf_exempt
def predict(request):
    
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)

    try:
        st_time = time.time()
        load_all_models()

        body = json.loads(request.body)
        image_data = body.get('image')
        selected_word = body.get('selected_word', '')
        hints = body.get('hints', [])

        if not image_data or not selected_word:
            return JsonResponse({'top_predictions': []})

        selected_word_normalized = selected_word.lower().replace(' ', '_')
        
        print(f"\n\nProcessing word: '{selected_word}' -> normalized: '{selected_word_normalized}'")
        def process_model(model,model_idx):
             
            print("\n 🔍 Processing model:", model_idx, "\n\n")
            
            # Get processed images
            processed_images = extract_and_resize_parts(image_data)
            image_b64_list = []
            visual_preds = []

            for img_tensor in processed_images:
                preds = model.predict(img_tensor, verbose=0)[0]
                top2_indices = np.argsort(preds)[::-1][:2]
                top2 = [(CLASSES[i].replace('_', ' ').title(), float(preds[i])) for i in top2_indices]
                visual_preds.append(top2)

                # Convert image tensor to base64
                img_arr = img_tensor[0, :, :, 0]
                img_b64 = image_to_base64(img_arr)
                image_b64_list.append(img_b64)
                
            start_time = time.time()
            # Get predictions from all processed images
            preds = [model.predict(img, verbose=0)[0] for img in processed_images]
            inference_time = time.time() - start_time
            # Extract top-N predictions
            N = 10
            top_pred_lists = [np.argsort(p)[::-1][:N] for p in preds]
            
            # Merge predictions by interleaving
            interleaved = []
            for i in range(N):
                for pred_set in top_pred_lists:
                    label = CLASSES[pred_set[i]]
                    if label not in interleaved:
                        interleaved.append(label)

            # Initialize session variables
            if 'previous_failed' not in request.session:
                request.session['previous_failed'] = []

            # Apply filtering with randomness
            if random.random() < 0.15:
                print("model",model_idx,": ⚠️ Skipping all filters due to randomness")
                filtered = interleaved
            elif random.random() < 0.20:
                print("model",model_idx,": ⚠️ Using only length-based filtering due to randomness")
                filtered = filter_by_length_only(interleaved, selected_word_normalized)
            else:
                print("model",model_idx,": Applying hints and length filtering")
                filtered = filter_predictions_by_hints(interleaved, selected_word_normalized, hints)

            # Convert to result format and calculate probabilities
            previous_failed = request.session.get('previous_failed', [])
            previous_failed_set = set(request.session.get('previous_failed', []))
            result = []
            
            
            for cls in filtered:
                cls_norm = cls.lower().replace(' ', '_')
                if cls_norm in previous_failed_set and cls_norm != selected_word_normalized:
                    continue  # skip known bad guesses
                prob = float(np.max([p[CLASSES.index(cls)] for p in preds]))
                display_name = cls.replace('_', ' ').title()
                result.append((display_name, prob))

            # Remove previously failed guesses (except correct answer)
            result = [
                (name, prob)
                for name, prob in result
                if name.lower().replace(' ', '_') not in previous_failed or 
                name.lower().replace(' ', '_') == selected_word_normalized
            ]
            random.shuffle(result)

            print("model",model_idx,f": Filtered predictions after removing failed: {result}")

            # Fallback with top 20 if no results
            if not result:
                if random.random() <0.5:
                    result=[(i,0) for i in interleaved[:N] if i[0].lower().replace(' ','_') not in previous_failed]
                else:
                    print("model",model_idx,": No results, trying fallback with top 20...")
                    N = 20
                    top_pred_lists = [np.argsort(p)[::-1][:N] for p in preds]
                    
                    interleaved = []
                    for i in range(N):
                        for pred_set in top_pred_lists:
                            label = CLASSES[pred_set[i]]
                            if label not in interleaved:
                                interleaved.append(label)

                    # Apply same filtering logic
                    if random.random() < 0.15:
                        filtered = interleaved
                    elif random.random() < 0.20:
                        
                        filtered = filter_by_length_only(interleaved, selected_word_normalized)
                    else:
                        filtered = filter_predictions_by_hints(interleaved, selected_word_normalized, hints)


                    previous_failed = request.session.get('previous_failed', [])
                    previous_failed_set = set(request.session.get('previous_failed', []))
                    result = []
                    

                    for cls in filtered:
                        cls_norm = cls.lower().replace(' ', '_')
                        if cls_norm in previous_failed_set and cls_norm != selected_word_normalized:
                            continue  # skip known bad guesses
                        prob = float(np.max([p[CLASSES.index(cls)] for p in preds]))
                        display_name = cls.replace('_', ' ').title()
                        result.append((display_name, prob))

                    result = [
                        (name, prob)
                        for name, prob in result
                        if name.lower().replace(' ', '_') not in previous_failed or 
                        name.lower().replace(' ', '_') == selected_word_normalized
                    ]
                    
                    #shuffle the result to add randomness
                    random.shuffle(result)
            
            return [
                result,
                inference_time,
                model_idx,  # Include model index for ordering
                [
                    {'img': b64, 'preds': preds}
                    for b64, preds in zip(image_b64_list, visual_preds)
                ]
            ]
        
        processed_images = extract_and_resize_parts(image_data)
        
        # --- DEBUG: Show image stats ---
        print("\n\n🖼️ Checking for blank canvas...\n\n")
        

        if all(is_blank(img) for img in processed_images):
            
            random.shuffle(RANDOM_GUESSES)
            random_guesses = [(name, 0.0) for name in RANDOM_GUESSES[:3]]
            model_count = len(loaded_models)
            model_order = list(range(model_count))
            random.shuffle(model_order)
            return JsonResponse({
                'top_predictions': [[random_guesses, 999.0, model_order[0], []]],
                'visual_inputs': [],
                'model_order': model_order,
                'delay_time': time.time() - st_time,
            })


      
        model_items = list(loaded_models.items())
        random.shuffle(model_items)
        shuffled_model_list = [(i, model_info['model']) for i, (_, model_info) in enumerate(model_items)]

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = {
        #         executor.submit(process_model, model, idx): idx
        #         for idx, model in shuffled_model_list
        #     }
        #     results = [future.result() for future in futures]
        
        def get_early_return_chance(num_models):
            """
            Calculate early return chance percentage based on number of models.
            
            Args:
                num_models (int): Number of parallel models
                
            Returns:
                float: Chance percentage (0-100)
                
            Criteria:
            - 1 model: 0%
            - 2 models: 15%
            - 4 models: 25%
            - 6 models: 35%
            - 8 models: 45%
            - 15+ models: 100%
            """
            if num_models <= 1:
                return 0.0
            elif num_models == 2:
                return 15.0
            elif num_models >= 15:
                return 100.0
            else:
                # Linear interpolation for values between known points
                # Using key points: (2,15), (4,25), (6,35), (8,45), (15,100)
                
                if num_models <= 4:
                    # Between 2 and 4: 15% to 25%
                    return 15.0 + (num_models - 2) * (25.0 - 15.0) / (4 - 2)
                elif num_models <= 6:
                    # Between 4 and 6: 25% to 35%
                    return 25.0 + (num_models - 4) * (35.0 - 25.0) / (6 - 4)
                elif num_models <= 8:
                    # Between 6 and 8: 35% to 45%
                    return 35.0 + (num_models - 6) * (45.0 - 35.0) / (8 - 6)
                else:
                    # Between 8 and 15: 45% to 100%
                    return 45.0 + (num_models - 8) * (100.0 - 45.0) / (15 - 8)

        num_models = len(shuffled_model_list)
        early_return_chance = get_early_return_chance(num_models)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        early_return_triggered = False

        with ThreadPoolExecutor() as executor:
            future_to_model = {
                executor.submit(process_model, model, idx): idx
                for idx, model in shuffled_model_list
            }

            for future in as_completed(future_to_model):
                try:
                    result = future.result()
                    results.append(result)

                    # 🎯 25% chance to return early
                    if not early_return_triggered and random.random() < (early_return_chance / 100.0):
                        early_return_triggered = True
                        print(f"\n\n⚠️ Returning early with model {result[2]}\n\n")
                        return JsonResponse({
                            'top_predictions': [result],
                            'visual_inputs': result[-1],
                            'model_order': [result[2]],
                            'delay_time': time.time() - st_time,
                        })

                except Exception as ex:
                    print(f"❌ Error getting result from model {future_to_model[future]}: {ex}")

        # Determine if any model guessed correctly
        any_correct = False
        for r in results:
            preds = r[0]  # r = [result, inference_time, model_idx, visual_inputs]
            if preds:
                top_guess = preds[0][0].lower().replace(' ', '_')
                if top_guess == selected_word_normalized:
                    any_correct = True
                    print(f"\n\n✅ Model {r[2]} guessed correctly: {top_guess}\n\n")
                    break

        if not any_correct:
            print(f"\n\n❌ No model guessed '{selected_word_normalized}' correctly.\n\n")
            guessed = results[0][0][0][0].lower().replace(' ', '_') if results[0][0] else "none"
            previous_failed = request.session.get('previous_failed', [])
            if guessed != selected_word_normalized and guessed not in previous_failed:
                previous_failed.append(guessed)
                request.session['previous_failed'] = previous_failed
                print(f"\n\n📝 Added to failed list: {guessed}\n\n")


        def top_label_confidence(res):
            preds = res[0]
            if preds:
                label = preds[0][0].lower().replace(' ', '_')
                confidence = preds[0][1]
                return label, confidence
            return None, 0.0

        # Check if all non-empty top labels are the same
        non_empty_results = [r for r in results if r[0]]
        top_labels = [top_label_confidence(r)[0] for r in non_empty_results]
        label_consensus = len(set(top_labels)) == 1 if top_labels else False

        if label_consensus:
            # If all top labels are same, sort by confidence (descending)
            results.sort(key=lambda r: top_label_confidence(r)[1], reverse=True)
            print(f"\n\n📊 Same top label '{top_labels[0]}' across models — sorted by confidence")
        else:
            # Default: sort by inference time (ascending)
            results.sort(key=lambda x: x[1])
            print("\n\n⚡ Different labels — sorted by inference time")

        model_order = [r[2] for r in results]

        print(f"Model order by inference time: {model_order}")
        print("Final results from all models:", [results[i][0] for i in range(len(results))])
        ed_time = time.time() - st_time
        return JsonResponse({
            'top_predictions': results,
            'visual_inputs': results[-1][-1],  # preds of final model only
            'model_order': model_order,
            'delay_time': ed_time,  # Return total processing time
        })

    except Exception as e:
        print("Prediction error:", e)
        return JsonResponse({'error': str(e)}, status=500)