import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import spacy
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gensim.downloader as api
import subprocess


subprocess.run(['pip', 'install', 'spacy'])
subprocess.run(['pip', 'install', 'nltk'])

subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Define the LLM class
class LLM:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_response(self, user_input, context=None):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        context_ids = self.tokenizer.encode(context, return_tensors="pt") if context else None

        if context_ids is not None:
            input_ids = torch.cat([context_ids, input_ids], dim=-1)

        response_ids = self.model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

class ChatBot:
    def __init__(self, dialog_data_file, product_data_file, use_llm=False):
        try:
            self.df = pd.read_csv(dialog_data_file)
            self.product_df = pd.read_csv(product_data_file)
        except FileNotFoundError:
            print("Error: Dialog or product data file not found.")
            self.df = None
            self.product_df = None
            return

        self.current_state = "Start"
        self.word2vec_model = api.load("word2vec-google-news-300")
        self.dialog_data_vectors = self.compute_dialog_data_vectors(self.df['user_prompt'].tolist())
        self.nlp = spacy.load("en_core_web_sm")
        self.use_llm = use_llm  
        if self.use_llm:
            self.llm = LLM()  

    def train(self, dialog_data_file):
        # Load your training data and add it to your existing dialog data.
        try:
            training_data = pd.read_csv(dialog_data_file)
            self.df = pd.concat([self.df, training_data])
            self.dialog_data_vectors = self.compute_dialog_data_vectors(self.df['user_prompt'].tolist())
            self.tfidf_matrix = self.tfidf_vectorizer.transform(self.df['user_prompt'].tolist())
            print("Chatbot has been trained with new data.")
        except FileNotFoundError:
            print("Error: Training data file not found.")

        self.conversation_context = []

        self.dialog_data = self.df['user_prompt'].tolist()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.dialog_data)

    def compute_dialog_data_vectors(self, dialog_data):
        return [self.compute_average_vector(self._preprocess_text(text)) for text in dialog_data]

    def _preprocess_text(self, text):
        return ' '.join([word for word in word_tokenize(text.lower()) if word not in stopwords.words('english')])

    def compute_average_vector(self, text):
        words = text.split()
        vectors = [self.word2vec_model[word] for word in words if word in self.word2vec_model]
        return sum(vectors) / len(vectors) if vectors else None

    def _intent_recognition(self, user_input):
        preprocessed_input = self._preprocess_text(user_input)

        product_match = self.product_df[
            self.product_df['Product Name'].str.lower() == preprocessed_input
        ]
        if not product_match.empty:

            product_info = product_match.iloc[0]
            response = f"You inquired about {product_info['Product Name']}. It is priced at {product_info['Price']} and is currently {product_info['Availability']}."
            return response, None

        similarity_scores = self.compute_semantic_similarity(preprocessed_input)

        most_similar_indices = similarity_scores.argsort()[0][::-1]
        best_match_score = similarity_scores[0, most_similar_indices[0]]

        if best_match_score > 0.5:
            best_match_index = most_similar_indices[0]
            return self.df.loc[best_match_index, 'system_reply'], self.df.loc[best_match_index, 'new_state']
        else:
            return None, None
        
    def respond_to_prompt(self, user_input):
        # Check if the user input contains an exit phrase
        if self._is_exit_phrase(user_input):
            return "Goodbye! If you have more questions, feel free to ask."

        # Determine the intent of the user input
        response, new_state = self._intent_recognition(user_input)

        if response is not None:
            # If the intent is recognized, return the appropriate response
            self.current_state = new_state if new_state else self.current_state
            return response

        # If no specific intent is recognized, generate a response using LLM
        if self.use_llm:
            context = ' '.join(self.conversation_context)
            response = self.llm.generate_response(user_input, context)
        else:
            response = "I'm sorry, but I didn't understand your question. Please try asking in a different way."

        # Update conversation context
        self.conversation_context.append(user_input)
        if len(self.conversation_context) > 2:
            self.conversation_context.pop(0)

        return response

    def compute_semantic_similarity(self, user_input):
        user_input_vector = self.compute_average_vector(user_input)
        similarity_scores = linear_kernel([user_input_vector], self.dialog_data_vectors)
        return similarity_scores

    def _dialog_manager(self, user_input):
        entities = self._entity_extraction(user_input)
        if 'product' in entities:
            self.current_state = "ProductInquiry" 
        else:
            self.current_state = "UserResponse"

    def _entity_extraction(self, user_input):
        doc = self.nlp(user_input)
        entities = {ent.label_: ent.text for ent in doc.ents}
        return entities

    def _is_exit_phrase(self, user_input):
        exit_phrases = ["goodbye", "thanks", "see you later", "bye", "that's all for now"]
        return any(fuzz.partial_ratio(phrase, user_input.lower()) >= 85 for phrase in exit_phrases)
