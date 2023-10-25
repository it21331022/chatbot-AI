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

    def respond_to_prompt(self, user_prompt):
        if self._is_exit_phrase(user_prompt):
            return "Goodbye! If you have more questions later, feel free to come back."


        response, new_state = self._intent_recognition(user_prompt)
        if new_state:
            self.current_state = new_state
        if response:
            self._dialog_manager(user_prompt)
            return response

        if self.current_state == "ProductInquiry":

            response = self.handle_product_inquiry(user_prompt)
            self.current_state = "UserResponse"  
            return response


        if self.use_llm and self.current_state == "UserResponse":

            conversation_context = self.get_conversation_context()


            llm_response = self.llm.generate_response(user_prompt, context=conversation_context)


            relevance_score = self.evaluate_response_relevance(llm_response, user_prompt)

            if relevance_score < 0.5:

                fallback_response = "I don't understand what you are saying. Did you mean #NEXT BEST OPTION#"
                self.append_to_conversation_context(user_prompt, fallback_response)
                return fallback_response


            self.append_to_conversation_context(user_prompt, llm_response)

            return llm_response

        return "I don't understand what you are saying."

    def evaluate_response_relevance(self, response, user_input):
        user_input_tfidf = self.tfidf_vectorizer.transform([user_input])
        response_tfidf = self.tfidf_vectorizer.transform([response])
        cosine_similarity_score = linear_kernel(user_input_tfidf, response_tfidf).flatten()[0]
        return cosine_similarity_score

    def get_conversation_context(self):
        return " ".join(self.conversation_context)

    def append_to_conversation_context(self, user_input, chatbot_response):

        self.conversation_context.append(user_input)
        self.conversation_context.append(chatbot_response)


    def handle_product_inquiry(self, user_input):
      preprocessed_input = self._preprocess_text(user_input)
      product_match = self.product_df[
          self.product_df['Product Name'].str.lower() == preprocessed_input
      ]
      if not product_match.empty:
          product_info = product_match.iloc[0]
          return f"{product_info['Product Name']}:\nDescription: {product_info['Product Description']}\nPrice: {product_info['Price']}\nBrand: {product_info['Brand']}\nModel: {product_info['Model']}\nAvailability: {product_info['Availability']}\nSpecifications: {product_info['Technical Specifications']}\nRatings and Reviews: {product_info['Ratings and Reviews']}"
      else:
          return "I'm sorry, I couldn't find information about that product."


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

if __name__ == "__main__":
    chatbot = ChatBot('dialog_data.csv', 'productdata.csv', use_llm=True)
    if chatbot.df is None or chatbot.product_df is None:
        exit()

    print("Hi, I'm your chatbot. You can start a conversation with me now.")

    while True:
        user_prompt = input()
        response = chatbot.respond_to_prompt(user_prompt)
        print(response)
        if "Goodbye!" in response:
            break