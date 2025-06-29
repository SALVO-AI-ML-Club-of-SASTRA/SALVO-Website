# Create your views here.
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os, random, json
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from tensorflow.keras.models import load_model
import base64
import matplotlib.pyplot as plt
import io
import time

# === Global Constants ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "step_176000.keras")

# Load model only once
model = load_model(MODEL_PATH)
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


def username_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        request.session['username'] = username
        return redirect('draw')
    return render(request, 'drawapp/username_page.html')


def draw_page(request):
    if 'username' not in request.session:
        return redirect('username')

    # Select 3 random classes for this session
    challenge_words = random.sample(CLASSES, 3)
    request.session['challenge_words'] = challenge_words

    # Pass readable names to the template
    readable_words = [w.replace('_', ' ').title() for w in challenge_words]

    return render(request, 'drawapp/draw_page.html', {
        'challenge_words': readable_words,
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
        crop = full_np[i * step:i * step + int(h / 2), i * step:i * step + int(w / 2)]
        crop = cv2.resize(crop, size)
        thirds.append(crop)

    # 3. Center crop
    ch, cw = 390, 390
    top = (h - ch) // 2
    left = (w - cw) // 2
    center_crop = full_np[top:top + ch, left:left + cw]
    center_resized = cv2.resize(center_crop, size)

    # 4. Optional 90-degree rotation on central 400x400 part
    rotation_aug = None
    if random.random() < rotation_prob:
        rot_ch, rot_cw = 400, 400
        rt = (h - rot_ch) // 2
        rl = (w - rot_cw) // 2
        rot_crop = full_np[rt:rt + rot_ch, rl:rl + rot_cw]
        rotated = cv2.rotate(rot_crop, cv2.ROTATE_90_CLOCKWISE)
        rotation_aug = cv2.resize(rotated, size)

    # Stack all for prediction
    all_images = [original] + thirds + [center_resized]
    if rotation_aug is not None:
        all_images.append(rotation_aug)

    all_images = [np.expand_dims(img, axis=(0, -1)) / 255.0 for img in all_images]  # shape: (1, 128, 128, 1)

    return all_images  # list of 5 image tensors + optional rotation augmentation


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
    print(f"Target word: '{target_word}' -> normalized: '{target_normalized}'")
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

    print(f"Final filtered results: {[normalize_word(w) for w in hint_filtered]}")
    return hint_filtered


def filter_by_length_only(candidates, target_word):
    """Filter candidates by length only (fallback function)"""

    def normalize_word(word):
        return word.replace('_', '').replace(' ', '').lower()

    target_length = len(normalize_word(target_word))
    return [c for c in candidates if len(normalize_word(c)) == target_length]


# Also update the predict function to use consistent normalization
@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)

    try:
        body = json.loads(request.body)
        image_data = body.get('image')
        selected_word = body.get('selected_word', '')
        hints = body.get('hints', [])

        # Normalize selected_word consistently
        # Frontend sends word with spaces, convert to our internal format
        selected_word_normalized = selected_word.lower().replace(' ', '_')

        if not image_data or not selected_word:
            return JsonResponse({'top_predictions': []})

        print(f"Processing word: '{selected_word}' -> normalized: '{selected_word_normalized}'")

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

        # Get predictions from all processed images
        preds = [model.predict(img, verbose=0)[0] for img in processed_images]

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
            print("⚠️ Skipping all filters due to randomness")
            filtered = interleaved
        elif random.random() < 0.20:
            print("⚠️ Using only length-based filtering due to randomness")
            filtered = filter_by_length_only(interleaved, selected_word_normalized)
        else:
            filtered = filter_predictions_by_hints(interleaved, selected_word_normalized, hints)

        # Convert to result format and calculate probabilities
        previous_failed = request.session.get('previous_failed', [])
        result = []

        for cls in filtered:
            # Calculate max probability across all prediction sets
            prob = float(np.max([p[CLASSES.index(cls)] for p in preds]))
            # Convert to display format (spaces instead of underscores, title case)
            display_name = cls.replace('_', ' ').title()
            result.append((display_name, prob))

        # Remove previously failed guesses (except correct answer)
        result = [
            (name, prob)
            for name, prob in result
            if name.lower().replace(' ', '_') not in previous_failed or
               name.lower().replace(' ', '_') == selected_word_normalized
        ]

        print(f"Filtered predictions after removing failed: {result}")

        # Fallback with top 20 if no results
        if not result:
            print("No results, trying fallback with top 20...")
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

            result = []
            for cls in filtered:
                prob = float(np.max([p[CLASSES.index(cls)] for p in preds]))
                display_name = cls.replace('_', ' ').title()
                result.append((display_name, prob))

            result = [
                (name, prob)
                for name, prob in result
                if name.lower().replace(' ', '_') not in previous_failed or
                   name.lower().replace(' ', '_') == selected_word_normalized
            ]

        # Track failed guesses
        if result:
            guessed = result[0][0].lower().replace(' ', '_')
            if guessed != selected_word_normalized:
                if guessed not in previous_failed:
                    previous_failed.append(guessed)
                request.session['previous_failed'] = previous_failed

        # Add delay
        if random.random() < 0.90:
            time.sleep(2.2)
        else:
            time.sleep(1.2)

        random.shuffle(result)
        print(f"Final result: {result}")

        return JsonResponse({
            'top_predictions': result,
            'visual_inputs': [
                {'img': b64, 'preds': preds}
                for b64, preds in zip(image_b64_list, visual_preds)
            ]
        })

    except Exception as e:
        print("Prediction error:", e)
        return JsonResponse({'error': str(e)}, status=500)