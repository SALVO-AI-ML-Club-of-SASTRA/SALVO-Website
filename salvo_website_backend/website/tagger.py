import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
from .tag_dataset import AIdict
import time
# Check if NLTK data directory exists, if not create it
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set NLTK data path to include current directory
nltk.data.path.append(nltk_data_dir)


try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    print("NLTK resources successfully downloaded.")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    print("Please run the following commands in your Python interpreter:")
    print(">>> import nltk")
    print(">>> nltk.download('punkt')")
    print(">>> nltk.download('stopwords')")
    print(">>> nltk.download('wordnet')")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class PostTagger:
    def __init__(self, label_dict, title_weight=2.0, max_tags=5, min_score=0.1):
        """
        Initialize the post tagger with a dictionary of labels and their associated keywords.
        
        Args:
            label_dict: Dictionary where keys are labels and values are lists of related words
            title_weight: How much more to weight the title compared to content (default: 2.0)
            max_tags: Maximum number of tags to return (default: 5)
            min_score: Minimum similarity score to consider a tag relevant (default: 0.1)
        """
        self.label_dict = label_dict
        self.title_weight = title_weight
        self.max_tags = max_tags
        self.min_score = min_score
        
        # Text processing tools
        
        self.stop_words = set(stopwords.words('english'))
       
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            print("Warning: WordNetLemmatizer not available, skipping lemmatization")
            self.lemmatizer = None
            
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Pre-process the label dictionary
        self.process_label_dict()
        
    def process_label_dict(self):
        """Pre-process label dictionary to create TF-IDF vectors for each label category"""
        # Combine all words for each label into a single string
        self.label_texts = {label: ' '.join(words) for label, words in self.label_dict.items()}
        
        # Create a corpus of all label texts
        corpus = list(self.label_texts.values())
        
        # Fit the vectorizer and transform the corpus
        self.label_vectors = self.vectorizer.fit_transform(corpus)
        
        # Store the labels in order for later reference
        self.labels = list(self.label_texts.keys())
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize (with fallback option)
        try:
            tokens = word_tokenize(text)    
        except:
            print("Warning: word_tokenize failed, using spacy's nlp ")
            tokens = text.split()
            

            # nlp = spacy.load("en_core_web_sm")
            # doc = nlp(text)
            # tokens = [token.text for token in doc]

        
        # Remove stopwords and lemmatize
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        else:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_word_frequency_score(self, post_text):
        """Calculate scores based on word frequency in post vs. label dictionaries"""
        # Clean and tokenize the post text
        clean_text = self.preprocess_text(post_text)
        tokens = clean_text.split()
        
        # Count word frequencies in the post
        word_counts = Counter(tokens)
        
        # Calculate scores for each label based on matching words
        scores = {}
        for label, words in self.label_dict.items():
            # Create a set of lowercase words for the label
            label_words = {word.lower() for word in words}
            
            # Calculate score as sum of frequencies of matching words
            score = sum(word_counts[word] for word in word_counts if word in label_words)
            
            # Normalize by number of words in the label dictionary
            if label_words:
                score /= len(label_words)
            
            scores[label] = score
        
        return scores
    
    def get_tfidf_similarity_score(self, post_text):
        """Calculate similarity scores based on TF-IDF vectors"""
        # Transform the post text using the fitted vectorizer
        post_vector = self.vectorizer.transform([post_text])
        
        # Calculate cosine similarity between post and each label
        similarities = cosine_similarity(post_vector, self.label_vectors).flatten()
        
        # Create a dictionary of scores
        scores = {self.labels[i]: similarities[i] for i in range(len(self.labels))}
        
        return scores
    
    def combine_scores(self, scores1, scores2, weight1=0.5, weight2=0.5):
        """Combine two score dictionaries with weights"""
        combined = {}
        all_labels = set(scores1.keys()) | set(scores2.keys())
        
        for label in all_labels:
            score1 = scores1.get(label, 0)
            score2 = scores2.get(label, 0)
            combined[label] = (score1 * weight1) + (score2 * weight2)
        
        return combined
    
    def tag_post(self, title, content=""):
        """
        Tag a post based on its title and content.
        
        Args:
            title: Post title
            content: Post content/description (optional)
        
        Returns:
            List of most relevant labels for the post
        """
        # Weight the title more heavily by repeating it
        weighted_title = " ".join([title] * int(self.title_weight))
        
        # Combine title and content
        full_text = f"{weighted_title} {content}"
        
        # Get scores using both methods
        freq_scores = self.get_word_frequency_score(full_text)
        tfidf_scores = self.get_tfidf_similarity_score(full_text)
        
        # Combine scores (60% TF-IDF, 40% frequency-based)
        combined_scores = self.combine_scores(freq_scores, tfidf_scores, 0.4, 0.6)
        best_label, best_score = max(combined_scores.items(), key=lambda x: x[1])
    
    # If the best score is too low, it's irrelevant
        if best_score < self.min_score:
            return ["other"]
        
        # Sort labels by score and filter out low scores
        sorted_labels = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_labels = [label for label, score in sorted_labels if score > self.min_score]
        
        # Return top labels up to max_tags
        return relevant_labels[:self.max_tags]