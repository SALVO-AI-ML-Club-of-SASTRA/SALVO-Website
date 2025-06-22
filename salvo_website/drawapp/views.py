
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

# === Global Constants ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "step_110000.keras")

#DATA_DIR = "/mnt/d/MyEverything/PythonProjects/Recent_projects/cnn_analysis/Hand_Drawing/quickdraw_images"  # Folder with 338 subfolders

# Load model only once
model = load_model(MODEL_PATH)
#CLASSES = sorted(os.listdir(DATA_DIR))  # class names = folder names
BASE_DIR = os.path.dirname(__file__)
CLASSES_PATH = os.path.join(BASE_DIR, "classes.txt")

with open(CLASSES_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
        

def draw_page(request):
    # Select 3 random classes for this session
    challenge_words = random.sample(CLASSES, 3)
    request.session['challenge_words'] = challenge_words

    # Pass readable names to the template
    readable_words = [w.replace('_', ' ').title() for w in challenge_words]

    return render(request, 'drawapp/draw_page.html', {
        'challenge_words': readable_words,
    })

def preprocess_base64_image(image_data):
    header, encoded = image_data.split(",", 1)
    image = Image.open(BytesIO(base64.b64decode(encoded))).convert('L')
    image = image.resize((128, 128))  # match model input
    img_arr = np.array(image).astype(np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=(0, -1))  # shape: (1, 128, 128, 1)
    return img_arr

def filter_predictions_by_hints(candidates, target_word, hints):
    filtered = [w for w in candidates if len(w) == len(target_word)]

    for hint in hints:
        index = hint.get('index')
        letter = hint.get('letter')
        filtered = [w for w in filtered if len(w) > index and w[index] == letter]

    return filtered

@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)

    try:
        body = json.loads(request.body)
        image_data = body.get('image')
        selected_word = body.get('selected_word', '').lower().replace(' ', '_')
        hints = body.get('hints', [])

        # Handle empty drawing
        if not image_data or not selected_word:
            return JsonResponse({'top_predictions': []})

        # Preprocess image
        image = preprocess_base64_image(image_data)

        # Run prediction
        predictions = model.predict(image, verbose=0)[0]  # shape: (338,)
        top_indices = np.argsort(predictions)[::-1][:15]
        top_classes = [CLASSES[i] for i in top_indices]
        print("Top classes:", top_classes)
        top_probs = [float(predictions[i]) for i in top_indices]

        # Filter with hints
        filtered_classes = filter_predictions_by_hints(top_classes, selected_word, hints)
        print("Filtered classes:", filtered_classes)
        # Prepare response
        # response = []
        # for cls in filtered_classes:
        #     idx = CLASSES.index(cls)
        #     response.append((cls.replace('_', ' ').title(), float(predictions[idx])))


        # return JsonResponse({'top_predictions': response})
        # Prepare all matching guesses with their confidence
        response = []
        for cls in filtered_classes:
            idx = CLASSES.index(cls)
            response.append((cls.replace('_', ' ').title(), float(predictions[idx])))

        # If multiple good candidates, shuffle them to inject randomness
        random.shuffle(response)
        print("Filtered predictions:", response)
        return JsonResponse({'top_predictions': response})


    except Exception as e:
        print("Prediction error:", e)
        return JsonResponse({'error': str(e)}, status=500)
