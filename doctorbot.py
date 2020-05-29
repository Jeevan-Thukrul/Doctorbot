# Importing necessary libraries and packages

import nltk
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

from gtts import gTTS

# Importing packages from nltk 

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

f = open('corpus.txt', 'r', errors='ignore')
raw = f.read()
# print(raw)

raw_ = raw.lower()
# print(raw_)

text = raw

# converts to list of sentences 
sent_tokens = nltk.sent_tokenize(text)
# print("Sent Tokens")
# print(sent_tokens)

# WordNet is a NLTK based dictionary of English
lemmer = nltk.stem.WordNetLemmatizer()


# Function to return a list of lemmatized lower case words and punctuations are removed
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# print("LemNormalise")
# print(LemNormalize(text))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    bot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        bot_response = bot_response + "I am sorry! I don't understand you"
        return bot_response
    else:
        bot_response = bot_response + sent_tokens[idx]
        return bot_response


def seperation():
    print()
    print("////////////////////////////////")
    print()


def audio_output(texto):
    tts = gTTS(texto)
    tts.save('/content/audio.mp3')
    # speak(texto)
    # os.system("mpg123 " + audio.mp3)


def response_generator(user_response):
    user_response = user_response.lower()

    if (user_response != 'exit'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            # print("Doctorbot: You are welcome...")
            return "You are welcome..."

        else:
            if (greeting(user_response) != None):
                # print("Doctorbot: "+ greeting(user_response))
                return greeting(user_response)

            else:
                # print("Doctorbot: ",end="")
                # print(response(user_response))
                return response(user_response)
                sent_tokens.remove(user_response)

    else:
        flag = False
        # print("Doctorbot: Bye! take care stay home stay safe")
        return "Bye! take care stay home stay safe"

# flag=True
# print("Doctorbot: My name is Doctorbot. I'm an informative Chatbot for Coronavirus. If you want to exit, type 'exit'")
# while(flag==True):
#     print("User:")
#     user_response = input()
#     #generator(user_response)
#     result = response_generator(user_response)
#     print (result)
