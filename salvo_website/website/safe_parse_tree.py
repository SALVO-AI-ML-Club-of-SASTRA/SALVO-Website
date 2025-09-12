import time
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Layer, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordfreq import zipf_frequency

def safety_check(input_text, OPTIMAL_LSTM_THRESHOLD=0.55,ENGLISH_RATIO_THRESHOLD=0.7,AI_LABEL_THRESHOLD=0.5,MAX_LSTM_INPUT_CONTEXT_WORD_LENGTH=50,MIN_LSTM_INPUT_CONTEXT_WORD_LENGTH=10):
    
    
    final_result={"check_english":{'status':"", 'ai_label_ratio':0},"check_direct_nsfw":{'nsfw_match_words':[],'nsfw_match_words_on_clean':[],'nsfw_match_words_on_ultra_clean':[],'status':"FAIL",},"check_lstm_attention_nsfw":{'status':"FAIL"},"overall_safety":"","guidelines":""}
    
    final_result={}
    def load_ai_labels(filepath="website\\regex_pattern.txt"):
        """Load AI labels from file into a set"""
        with open(filepath, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())
        
    
    
    def is_english_word(word, threshold=-1):
        """
        Check if a word is English using wordfreq.
        zipf_frequency returns a log frequency value.
        Common words ~ 6–7, rare but valid words ~ 2–3, nonsense ~ 0.
        """
        return zipf_frequency(word, "en") > 0  # accept any recognized word

    def check_paragraph(paragraph, ai_labels):
        # Clean and tokenize (letters + numbers only)
        tokens = re.findall(r'\b\w+\b', paragraph.lower())

        if not tokens:
            return ""  # No valid words

        english_count = sum(1 for w in tokens if is_english_word(w))
        non_english = [w for w in tokens if not is_english_word(w)]
        english = [w for w in tokens if is_english_word(w)]
        total = len(tokens)
        english_ratio = english_count / total
        ai_label_count = sum(1 for w in english if w in ai_labels)
        ai_label_ratio = ai_label_count / len(english) if english else 0
        ai_labels_found = [w for w in english if w in ai_labels]

        # Case 1: ≥70% English
        if english_ratio >= ENGLISH_RATIO_THRESHOLD:
            return {
                "status": "PASS",
                "paragraph": paragraph,
                "total_words": total,
                "english_words": english_count,
                "english_ratio": round(english_ratio * 100, 2),
                "ai_label_ratio": round(ai_label_ratio * 100, 2),
                "non_english_words": non_english,
                "ai_labels_found": ai_labels_found,
            }

        # Case 2: Check non-English words against AI labels
        if non_english and ai_label_ratio >= AI_LABEL_THRESHOLD:
            return {
                "status": "PASS",
                "paragraph": paragraph,
                "total_words": total,
                "english_words": english_count,
                "english_ratio": round(english_ratio * 100, 2),
                "ai_label_ratio": round(ai_label_ratio * 100, 2),
                "ai_labels_found": ai_labels_found,
                "non_english_words": non_english,
            }

        # FAIL
        return {
            "status": "FAIL",
            "paragraph": paragraph,
            "total_words": total,
            "english_words": english_count,
            "english_ratio": round(english_ratio * 100, 2),
            "ai_label_ratio": round(ai_label_ratio * 100, 2),
            "ai_labels_found": ai_labels_found,
            "non_english_words": non_english,
        }
    
    
    
    ai_labels = load_ai_labels("website\\unique_ai_labels.txt")
    check1_result = check_paragraph(input_text, ai_labels)
    
    final_result["check_english"]=check1_result
    
    if final_result["check_english"]["status"]=="PASS":
        
        import string

        # Mapping of leetspeak/symbols to normal chars
        leet_map = {
            '@': 'a',
            '8': 'b',
            '3': 'e',
            '6': 'g',
            '1': 'i',
            '!': 'i',
            '|': 'i',
            '0': 'o',
            '$': 's',
            '5': 's',
            '+': 't',
            '2': 'z',
        }

        def normalize_leet(text):
            """Replace leetspeak characters with their normal equivalents."""
            for k, v in leet_map.items():
                text = text.replace(k, v)
            return text

        def clean_word_basic(word):
            """Lowercase and remove punctuation (no leet normalization)."""
            return word.lower().translate(str.maketrans("", "", string.punctuation))

        def clean_word_leet(word):
            """Lowercase, normalize leetspeak, then remove punctuation."""
            word = word.lower()
            word = normalize_leet(word)           # convert @ -> a, 0 -> o, etc.
            word = word.translate(str.maketrans("", "", string.punctuation))  
            return word


        def load_pattern(filename):
            with open(filename, "r", encoding="utf-8") as f:
                regex = f.read().strip()
            return re.compile(regex, re.IGNORECASE)

        def check_message(msg, pattern):
            words = msg.split()

            # Three cleaning methods
            basic_cleaned = [clean_word_basic(w) for w in words]
            leet_cleaned = [clean_word_leet(w) for w in words]
            leet_on_basic = [normalize_leet(b) for b in basic_cleaned]

            nsfw_matches = []
            possible_nsfw_matches = []
            possible_nsfw_from_basic = []

            for orig, basic, leet, leet_b in zip(words, basic_cleaned, leet_cleaned, leet_on_basic):
                #print(f"orig={orig}, basic={basic}, leet={leet}, leet_b={leet_b}")
                # Check with basic cleaning (no leet)
                if pattern.search(basic):
                    nsfw_matches.append(orig)

                # Check with leet normalization (direct path)
                if pattern.search(leet):
                    possible_nsfw_matches.append(orig)

                # Check with leet applied on already basic-cleaned word
                if pattern.search(leet_b):
                    possible_nsfw_from_basic.append(orig)

            if nsfw_matches or possible_nsfw_matches or possible_nsfw_from_basic:
                return "NSFW/Vulgar", nsfw_matches, possible_nsfw_matches, possible_nsfw_from_basic
            else:
                return "SFW", [], [], []
            
        pattern_reg = load_pattern("website\\regex_pattern.txt")
        check2_result_tuple=check_message(input_text, pattern_reg)
        check2_result={
            "status": check2_result_tuple[0],
            "nsfw_match_words": check2_result_tuple[1],
            "nsfw_match_words_on_clean": check2_result_tuple[2],
            "nsfw_match_words_on_ultra_clean": check2_result_tuple[3],
        }
        final_result["check_direct_nsfw"]=check2_result
        
        if final_result["check_direct_nsfw"]["status"]=="SFW":
        
            #check 3: LSTM with attention model classifier
            class HateSpeechDetector:
                def __init__(self, max_words=10000, max_len=100, embedding_dim=100):
                    self.max_words = max_words
                    self.max_len = max_len
                    self.embedding_dim = embedding_dim
                    self.tokenizer = None
                    self.model = None
                    self.scaler = None
                    
                def clean_text(self, text):
                    """Clean and preprocess text"""
                    if pd.isna(text):
                        return ""
                    text = str(text).lower()
                    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
                    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
                    text = re.sub(r"&amp;\w+;", "", text)  # Remove HTML entities
                    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only alphabetic chars
                    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
                    return text
                    
                def prepare_data(self, df):
                    """Prepare and transform the dataset"""
                    print("Cleaning text data...")
                    df['clean_text'] = df['tweet'].apply(self.clean_text)
                    
                    # Create binary target: hate/offensive (1) vs neither (0)
                    df['hate_offensive'] = np.where(df['class'] == 2, 0, 1)
                    
                    # Normalize ratings to 1-10 scale
                    self.scaler = MinMaxScaler(feature_range=(1, 10))
                    
                    # Combine hate_speech and offensive_language ratings for hate/offensive class
                    # Use 'neither' rating for neither class
                    df['rating_hate_offensive'] = df[['hate_speech', 'offensive_language']].max(axis=1)
                    
                    # Scale ratings
                    hate_off_scaled = self.scaler.fit_transform(df['rating_hate_offensive'].values.reshape(-1, 1))
                    neither_scaled = self.scaler.fit_transform(df['neither'].values.reshape(-1, 1))
                    
                    df['rating_hate_offensive_scaled'] = hate_off_scaled.flatten()
                    df['rating_neither_scaled'] = neither_scaled.flatten()
                    
                    # Prepare features - ONLY TEXT
                    X_text = df['clean_text'].values
                    y_class = df['hate_offensive'].values
                    
                    # Create rating targets based on class
                    y_rating = np.where(df['hate_offensive'] == 1, 
                                    df['rating_hate_offensive_scaled'], 
                                    df['rating_neither_scaled'])
                    
                    return X_text, y_class, y_rating
                    
                def prepare_tokenizer(self, texts):
                    """Prepare text tokenizer"""
                    self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
                    self.tokenizer.fit_on_texts(texts)
                    
                    # Convert text to sequences
                    sequences = self.tokenizer.texts_to_sequences(texts)
                    X_pad = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
                    
                    return X_pad

                
                class AttentionLayer(Layer):
                    """Custom Attention Layer with optional attention return"""
                    def __init__(self, return_attention=False, **kwargs):
                        super(HateSpeechDetector.AttentionLayer, self).__init__(**kwargs)
                        self.return_attention = return_attention
                
                    def build(self, input_shape):
                        self.W = self.add_weight(name="att_weight", 
                                                shape=(input_shape[-1], 1), 
                                                initializer="random_normal", 
                                                trainable=True)
                        self.b = self.add_weight(name="att_bias", 
                                                shape=(input_shape[1], 1), 
                                                initializer="zeros", 
                                                trainable=True)
                        super(HateSpeechDetector.AttentionLayer, self).build(input_shape)
                
                    def call(self, x):
                        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
                        a = tf.keras.backend.softmax(e, axis=1)   # attention weights
                        output = x * a
                        context = tf.keras.backend.sum(output, axis=1)
                        if self.return_attention:
                            return [context, a]   # return both context & weights
                        return context

                def compute_output_shape(self, input_shape):
                    if self.return_attention:
                        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]
                    return (input_shape[0], input_shape[-1])

                    
                def build_model(self):
                    """Build LSTM with Attention Multi-Output Model - TEXT INPUT ONLY"""
                    # Text input only
                    text_input = Input(shape=(self.max_len,), name='text_input')
                    embedding = Embedding(input_dim=self.max_words, 
                                        output_dim=self.embedding_dim, 
                                        input_length=self.max_len, 
                                        mask_zero=False)(text_input)
                    
                    # Bidirectional LSTM with attention
                    bi_lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(embedding)
                    #attention = self.AttentionLayer()(bi_lstm)
                    attention, att_weights = self.AttentionLayer(return_attention=True)(bi_lstm)

                    # Dense layers
                    dense1 = Dense(256, activation='relu')(attention)
                    dropout1 = Dropout(0.5)(dense1)
                    dense2 = Dense(128, activation='relu')(dropout1)
                    dropout2 = Dropout(0.4)(dense2)
                    dense3 = Dense(64, activation='relu')(dropout2)
                    dropout3 = Dropout(0.3)(dense3)
                    
                    # Multi-output heads
                    # Binary classification with sigmoid
                    class_output = Dense(1, activation='sigmoid', name='class_output')(dropout3)
                    
                    # Rating prediction (1-10 scale)
                    rating_output = Dense(1, activation='linear', name='rating_output')(dropout3)
                    
                    # Create model with only text input
                    #self.model = Model(inputs=text_input, outputs=[class_output, rating_output])
                    self.model = Model(inputs=text_input, 
                            outputs=[class_output, rating_output])
                    self.attention_model = Model(inputs=text_input, outputs=att_weights)


                    return self.model

                def train(self, X_text, y_class, y_rating, test_size=0.2):
                    """Train the model"""
                    print("Preparing text sequences...")
                    X_pad = self.prepare_tokenizer(X_text)
                    
                    
                    # Train-test split with stratification
                    X_train_text, X_test_text, y_train_class, y_test_class, y_train_rating, y_test_rating = train_test_split(
                        X_pad, y_class, y_rating, 
                        test_size=test_size, random_state=42, stratify=y_class
                    )
                    
                    print("Building model...")
                    self.build_model()
                    
                    # Calculate class weights for imbalanced dataset
                    class_weights = class_weight.compute_class_weight(
                        'balanced', classes=np.unique(y_train_class), y=y_train_class
                    )
                    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
                    
                    print(f"Class distribution: {np.bincount(y_train_class)}")
                    print(f"Class weights: {class_weight_dict}")
                    
                    # Compile model with focal loss for class imbalance
                    def focal_loss(alpha=0.25, gamma=2.0):
                        def focal_loss_fixed(y_true, y_pred):
                            y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
                            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
                            return -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
                        return focal_loss_fixed

                    def weighted_focal_loss(class_weights, alpha=0.25, gamma=2.0):
                        def focal_loss_fixed(y_true, y_pred):
                            y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
                            y_true_int = tf.cast(y_true, tf.int32)
                            
                            # Apply class weights
                            weights = tf.gather(list(class_weights.values()), y_true_int)
                            
                            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
                            
                            # Include class weights in the loss
                            return -weights * alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
                        return focal_loss_fixed

                    self.model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss={'class_output': weighted_focal_loss(class_weight_dict,alpha=0.75, gamma=2.0), 'rating_output': 'mse'},
                        metrics={'class_output': ['accuracy'], 'rating_output': ['mae']},
                        loss_weights={'class_output': 1.5, 'rating_output': 0.3}
                    )
                    
                    # Callbacks
                    early_stopping = EarlyStopping(
                        monitor="val_class_output_accuracy", mode='max', patience=5, 
                        restore_best_weights=True, verbose=1
                    )

                    lr_scheduler = ReduceLROnPlateau(
                        monitor='val_class_output_accuracy',
                        mode='max',
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=1
                    )

                    model_checkpoint = ModelCheckpoint(
                        'best_hate_speech_model3.keras', save_best_only=True, 
                        monitor='val_class_output_accuracy', mode='max', verbose=1
                    )
                    
                    print("Training model...")
                    # Train model with class weights - only text input
                    history = self.model.fit(
                        X_train_text,  # Only text input
                        [y_train_class, y_train_rating],
                        validation_data=(X_test_text, [y_test_class, y_test_rating]),  # Only text input for validation
                        epochs=30,
                        batch_size=64,
                        callbacks=[early_stopping, model_checkpoint, lr_scheduler],
                        verbose=1
                    )
                    
                    # Store test data for evaluation
                    self.X_test_text = X_test_text
                    self.y_test_class = y_test_class
                    self.y_test_rating = y_test_rating
                    
                    return history
                    
                def evaluate(self):
                    """Evaluate the model"""
                    if self.model is None:
                        print("Model not trained yet!")
                        return
                    
                    # Predictions - only text input
                    predictions = self.model.predict(self.X_test_text)
                    y_pred_class = (predictions[0] > 0.5).astype(int).flatten()
                    y_pred_rating = predictions[1].flatten()
                    
                    # Classification metrics
                    print("=== CLASSIFICATION RESULTS ===")
                    print(f"Accuracy: {np.mean(y_pred_class == self.y_test_class):.4f}")
                    print("\nClassification Report:")
                    print(classification_report(self.y_test_class, y_pred_class, 
                                            target_names=['Neither', 'Hate/Offensive']))
                    
                    # Confusion Matrix
                    cm = confusion_matrix(self.y_test_class, y_pred_class)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Neither', 'Hate/Offensive'],
                            yticklabels=['Neither', 'Hate/Offensive'])
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.show()
                    
                    # Rating prediction metrics
                    print("\n=== RATING PREDICTION RESULTS ===")
                    mae = np.mean(np.abs(y_pred_rating - self.y_test_rating))
                    rmse = np.sqrt(np.mean((y_pred_rating - self.y_test_rating)**2))
                    print(f"Mean Absolute Error: {mae:.4f}")
                    print(f"Root Mean Square Error: {rmse:.4f}")
                    
                    # Print prediction distribution
                    print(f"\nPrediction Distribution:")
                    print(f"Predicted Neither: {np.sum(y_pred_class == 0)}")
                    print(f"Predicted Hate/Offensive: {np.sum(y_pred_class == 1)}")
                    print(f"Actual Neither: {np.sum(self.y_test_class == 0)}")
                    print(f"Actual Hate/Offensive: {np.sum(self.y_test_class == 1)}")
                    
                    return y_pred_class, y_pred_rating
                    
                def predict_text(self, text):
                    """Predict on new text input"""
                    if self.model is None or self.tokenizer is None:
                        print("Model not trained yet!")
                        return None, None
                    
                    # Clean text
                    clean = self.clean_text(text)
                    
                    # Convert to sequence
                    sequence = self.tokenizer.texts_to_sequences([clean])
                    padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
                    
                    # Predict - only text input
                    predictions = self.model.predict(padded, verbose=0)
                    class_pred = predictions[0][0][0]
                    rating_pred = predictions[1][0][0]
                    
                    # Determine safety
                    is_safe = class_pred < OPTIMAL_LSTM_THRESHOLD
                    safety_label = "SAFE" if is_safe else "UNSAFE"
                    confidence = (1 - class_pred) if is_safe else class_pred
                    
                    return {
                        'text': text,
                        'safety': safety_label,
                        'confidence': confidence,
                        'toxicity_score': rating_pred,
                        'raw_probability': class_pred
                    }


                    

            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from collections import defaultdict

            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords

            #stop_words = set(stopwords.words('english'))


            def interactive_test_paragraph(model, attention_model, tokenizer, text):
                import re

                # NLTK stop words
                stop_words = set(stopwords.words('english'))

                sentences = re.split(r'\.|\n', text)
                processed_sentences = []

                # Split long sentences
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    words = s.split()
                    while len(words) > MAX_LSTM_INPUT_CONTEXT_WORD_LENGTH:
                        mid = len(words) // 2
                        processed_sentences.append(' '.join(words[:mid]))
                        words = words[mid:]
                    processed_sentences.append(' '.join(words))

                # Keep sentences >= 10 words
                final_sentences = [s for s in processed_sentences if len(s.split()) >= MIN_LSTM_INPUT_CONTEXT_WORD_LENGTH]

                if not final_sentences:
                    print("No valid sentences to process.")
                    return {
                    #"text": text,
                    "status": "FAIL",
                    "avg_confidence": 0,
                    "avg_score":0,
                    "avg_raw_prob": 0,
                    "top_influential_words": [],
                    "top_unique_influential_words": [],
                    "processed_sentences": []
                    }

                all_sentence_scores = []
                all_sentence_confidences = []
                all_sentence_raw = []
                sentence_safe_flags = []
                word_attention_totals = defaultdict(float)

                for s in final_sentences:
                    seq = tokenizer.texts_to_sequences([s])
                    padded = pad_sequences(seq, maxlen=100, padding="post", truncating="post")

                    predictions = model.predict(padded, verbose=0)
                    class_pred = predictions[0][0][0]
                    rating_pred = predictions[1][0][0]

                    att_weights = attention_model.predict(padded, verbose=0)[0].flatten()
                    seq_ids = padded[0]
                    words = [tokenizer.index_word.get(i, "<PAD>") for i in seq_ids if i != 0]
                    weights = att_weights[:len(words)]

                    # Top 6 words for this sentence
                    word_influence = list(zip(words, weights))
                    top_words = sorted(word_influence, key=lambda x: x[1], reverse=True)[:6]

                    # Add attention to total per word
                    for w, wt in top_words:
                        word_attention_totals[w] += wt

                    # Safety and confidence
                    is_safe = class_pred < OPTIMAL_LSTM_THRESHOLD
                    confidence = (1 - class_pred) if is_safe else class_pred

                    all_sentence_scores.append(rating_pred)
                    all_sentence_confidences.append(confidence)
                    all_sentence_raw.append(class_pred)
                    sentence_safe_flags.append(is_safe)

                # Determine overall safety
                if any(not safe for safe in sentence_safe_flags):
                    overall_safety_label = "UNSAFE"
                    scores_to_avg = [s for s, safe in zip(all_sentence_scores, sentence_safe_flags) if not safe]
                    avg_confidence = sum([c for c, safe in zip(all_sentence_confidences, sentence_safe_flags) if not safe]) / len(scores_to_avg)
                    avg_score = sum(scores_to_avg) / len(scores_to_avg)
                    avg_raw_probability = sum([r for r, safe in zip(all_sentence_raw, sentence_safe_flags) if not safe]) / len(scores_to_avg)
                else:
                    overall_safety_label = "SAFE"
                    scores_to_avg = [s for s, safe in zip(all_sentence_scores, sentence_safe_flags) if safe]
                    avg_confidence = sum([c for c, safe in zip(all_sentence_confidences, sentence_safe_flags) if safe]) / len(scores_to_avg)
                    avg_score = sum(scores_to_avg) / len(scores_to_avg)
                    avg_raw_probability = sum([r for r, safe in zip(all_sentence_raw, sentence_safe_flags) if safe]) / len(scores_to_avg)

                # Sort unique top words by total attention value (with stop words)
                top_words_with_stopwords = [w for w, wt in sorted(word_attention_totals.items(), key=lambda x: x[1], reverse=True)]

                # Sort unique top words by total attention value (without stop words)
                top_words_without_stopwords = [w for w, wt in sorted(word_attention_totals.items(), key=lambda x: x[1], reverse=True)
                                            if w.lower() not in stop_words]

                result = {
                    #"text": text,
                    "status": overall_safety_label,
                    "avg_confidence": float(avg_confidence),
                    "avg_score": float(avg_score),
                    "avg_raw_prob": float(avg_raw_probability),
                    "top_influential_words": top_words_with_stopwords,
                    "top_unique_influential_words": top_words_without_stopwords,
                    "processed_sentences": final_sentences
                }

                # # Print summary
                # print(
                #     'Overall Safety:', overall_safety_label,
                #     '\nAverage Confidence:', float(avg_confidence),
                #     '\nAverage Score:', float(avg_score),
                #     '\nAverage Raw Probability:', float(avg_raw_probability),
                #     '\nTop Influential Words (with stopwords):', top_words_with_stopwords,
                #     '\nTop Influential Words (without stopwords):', top_words_without_stopwords
                # )

                return result

                    
            def identify_oov_words(text, tokenizer, max_len=100):
                """
                Identify out-of-vocabulary words in the input text
                
                Args:
                    text (str): Input text to analyze
                    tokenizer: Fitted Keras tokenizer
                    max_len (int): Maximum sequence length
                
                Returns:
                    dict: Contains original words, their positions, and OOV information
                """
                import re
                
                # Clean text the same way as in your model
                def clean_text(text):
                    if pd.isna(text):
                        return ""
                    text = str(text).lower()
                    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
                    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
                    text = re.sub(r"&amp;\w+;", "", text)  # Remove HTML entities
                    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only alphabetic chars
                    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
                    return text
                
                # Clean the text
                cleaned_text = clean_text(text)
                
                # Split into words
                words = cleaned_text.split()
                
                # Get tokenizer vocabulary
                word_index = tokenizer.word_index
                index_word = tokenizer.index_word
                
                # Find OOV token ID
                oov_token_id = tokenizer.word_index.get(tokenizer.oov_token, 1) if hasattr(tokenizer, 'oov_token') and tokenizer.oov_token else 1
                
                # Tokenize the text
                sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
                padded = pad_sequences([sequence], maxlen=max_len, padding='post', truncating='post')[0]
                
                # Identify OOV words
                oov_words = []
                in_vocab_words = []
                
                for i, word in enumerate(words):
                    if word in word_index:
                        in_vocab_words.append({
                            'word': word,
                            'position': i,
                            'token_id': word_index[word],
                            'is_oov': False
                        })
                    else:
                        oov_words.append({
                            'word': word,
                            'position': i,
                            'token_id': oov_token_id,
                            'is_oov': True
                        })
                
                # Map sequence positions to words
                sequence_analysis = []
                word_idx = 0
                for i, token_id in enumerate(padded):
                    if token_id == 0:  # Padding
                        sequence_analysis.append({
                            'sequence_pos': i,
                            'token_id': token_id,
                            'word': '<PAD>',
                            'is_oov': False,
                            'is_padding': True
                        })
                    elif word_idx < len(words):
                        word = words[word_idx] if word_idx < len(words) else '<UNK>'
                        is_oov = token_id == oov_token_id
                        sequence_analysis.append({
                            'sequence_pos': i,
                            'token_id': token_id,
                            'word': word if not is_oov else f"{word} (OOV)",
                            'is_oov': is_oov,
                            'is_padding': False
                        })
                        word_idx += 1
                    else:
                        sequence_analysis.append({
                            'sequence_pos': i,
                            'token_id': token_id,
                            'word': index_word.get(token_id, '<UNK>'),
                            'is_oov': token_id == oov_token_id,
                            'is_padding': False
                        })
                
                return {
                    'original_text': text,
                    'cleaned_text': cleaned_text,
                    'original_words': words,
                    'oov_words': oov_words,
                    'in_vocab_words': in_vocab_words,
                    'sequence_analysis': sequence_analysis,
                    'oov_count': len(oov_words),
                    'total_words': len(words),
                    'oov_percentage': (len(oov_words) / len(words) * 100) if words else 0,
                    'tokenizer_vocab_size': len(word_index)
                }

            def print_oov_analysis(text, tokenizer, max_len=100):
                """
                Print a detailed analysis of OOV words in the text
                """
                analysis = identify_oov_words(text, tokenizer, max_len)
                
                # print(f"=== OOV ANALYSIS ===")
                # print(f"Original text: {analysis['original_text']}")
                # print(f"Cleaned text: {analysis['cleaned_text']}")
                # print(f"Total words: {analysis['total_words']}")
                # print(f"OOV words: {analysis['oov_count']}")
                # print(f"OOV percentage: {analysis['oov_percentage']:.1f}%")
                # print(f"Tokenizer vocabulary size: {analysis['tokenizer_vocab_size']}")
                
                # if analysis['oov_words']:
                #     print(f"\n=== OUT-OF-VOCABULARY WORDS ===")
                #     for oov_word in analysis['oov_words']:
                #         print(f"  • '{oov_word['word']}' at position {oov_word['position']}")
                # else:
                #     print(f"\n✅ No OOV words found!")
                
                # print(f"\n=== SEQUENCE BREAKDOWN ===")
                OOV_words=set()
                for item in analysis['sequence_analysis'][:15]:  # Show first 15 tokens
                    status = "OOV" if item['is_oov'] else ("PAD" if item['is_padding'] else "✓")
                    #print(f"  {item['sequence_pos']:2d}: {item['token_id']:4d} -> '{item['word']:<15}' [{status}]")
                    if status == "OOV":
                        OOV_words.add("".join(item['word'].split("(OOV)")))
                        
                return OOV_words
                        

                    
            import pickle
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.models import load_model
            def load_model_method2():
                """Load model by accessing AttentionLayer from HateSpeechDetector class"""
                
                # Create a temporary instance to access the nested class
                temp_detector = HateSpeechDetector()
                
                custom_objects = {
                    'AttentionLayer': temp_detector.AttentionLayer,
                    'HateSpeechDetector>AttentionLayer': temp_detector.AttentionLayer,
                
                }
                
                try:
                    model = load_model("website\\best_hate_speech_model3.keras", custom_objects=custom_objects,compile=False)
                    attention_model = load_model("website\\LSTM_attention_model3.keras", 
                                            custom_objects=custom_objects, 
                                            compile=False)
                    print("Model and attention model loaded successfully")
                    return model,attention_model
                except Exception as e:
                    print(f"Method 2 failed: {e}")
                    return None
                
            model,attention_model= load_model_method2()



            # Load tokenizer
            with open("website\\tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
                print("✅ Tokenizer loaded successfully!")
                
            check3_result={} if not interactive_test_paragraph(model, attention_model, tokenizer, input_text) else interactive_test_paragraph(model, attention_model, tokenizer, input_text)
            oov_words=print_oov_analysis(input_text, tokenizer, max_len=100)
            
            check3_result["oov_words"]= [] if not oov_words else list(oov_words)
            
            final_result["check_lstm_attention_nsfw"]=check3_result
            
            if final_result["check_lstm_attention_nsfw"]["status"]=="SAFE":
                final_result["overall_safety"]="SAFE"
                final_result["guidelines"]= "Content appears safe based on LSTM with Attention model analysis. No offensive or harmful language detected."
                return final_result
            
            else:
                final_result["overall_safety"]="1/3 UNSAFE"
                final_result["guidelines"]= "Content flagged as unsafe by LSTM with Attention model. Please review for potential offensive or harmful language."
                return final_result
            
        else:
            final_result["overall_safety"]="2/3 UNSAFE"
            final_result["guidelines"]= "Content flagged as unsafe by direct NSFW/Offensive keyword check. Please review for potential offensive or harmful language."
            return final_result
        
    else:
        final_result["overall_safety"]="3/3 UNSAFE"
        final_result["guidelines"]= "Content flagged as unsafe by AI-related keyword density and English language checks. Please review for potential offensive or harmful language or Non-English unprofessional language."
        return final_result
    

if __name__ == "__main__":
    import time
    test_text = """AI for Climate Change: Smarter Solutions for a Sustainable Future

Artificial Intelligence is no longer limited to chatbots and image recognition—it’s playing a vital role in tackling climate change. By combining machine learning, IoT sensors, and predictive modeling, AI systems can help reduce carbon emissions and improve energy efficiency.

⚡ Key Applications

Smart Energy Grids: AI predicts power demand, optimizes renewable integration, and reduces wastage.

Climate Modeling: Deep learning enhances weather predictions, helping governments prepare for extreme events.

Sustainable Agriculture: AI-driven systems monitor soil health, predict crop yields, and reduce fertilizer overuse.

Carbon Capture & Monitoring: AI models analyze satellite imagery to track deforestation and carbon emissions in real-time.

🌱 Why it Matters
With global temperatures rising, AI provides actionable insights to accelerate the shift toward sustainability. It’s not just about data—it’s about transforming industries to build a greener, smarter planet."""
  
    start_time= time.time()
    result = safety_check(test_text)
    end_time=time.time()
    print(f"Time taken: {end_time-start_time:.2f} seconds")
    print("\n",result)

    test_text = """AI? More Like Artificial Idiot

Everyone keeps worshipping AI like it’s some god, but let’s be real:

Half these “AI chatbots” are just glorified parrots that spit back nonsense.

Image generators? Cool—if you like mutant hands and faces that look like cursed nightmares.

And don’t get me started on “AI ethics committees”… bunch of clowns pretending they can leash a beast they don’t even understand.

AI isn’t going to replace humans anytime soon—it can’t even write a simple essay without making stuff up. But hey, keep hyping it up, 
Silicon Valley bros. Maybe one day your “intelligent system” will finally learn how not to be a lying piece of trash."""
    start_time= time.time()
    result = safety_check(test_text)
    end_time=time.time()
    print(f"Time taken: {end_time-start_time:.2f} seconds")
    print("\n",result)
        
            
            
    
    
    
    
    