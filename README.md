# pos_tag
 I create my own model for pos_tag like nltk and spacy have.
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import speech_recognition as sr
from sklearn.model_selection import train_test_split
import pyttsx3

# Load the Spacy language model
nlp = spacy.load('en_core_web_sm')


# Define the text to be tokenized
text = """
This is an example sentence for tokenization.
Hello how are you ? tell me what to do.
by the way nice to meet you.
what is your name ? I like apple.Then, you can modify your code to use speech recognition.
Here's an example of how you can do this.Decision trees are represented as tree structures, where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents a prediction.
The algorithm works by recursively splitting the data into smaller and smaller subsets based on the feature values.
At each node, the algorithm chooses the feature that best splits the data into groups with different target values.
To divide the data into subsets that are as pure as possible about the target variable, the tree is built recursively, beginning at the root node and selecting the most informative characteristic.
The aforementioned procedure persists until a halting condition is fulfilled, generally at attaining a specific depth or upon the node possessing a minimum quantity of data points.
Decision trees are a good tool for elucidating the logic behind forecasts since they are simple to see and comprehend.
CART can be used to create regression trees for continuous target variables in addition to classification.
When deciding which subsets to split, the algorithm in this instance minimizes the variance of the target variable inside each subset.
The datasets have both numerical and categorical features.Categorical features refer to string data types and can be easily understood by human beings.
However, machines cannot interpret the categorical data directly. Therefore, the categorical data must be converted into numerical data for further processing.
Don't miss your chance to ride the wave of the data revolution! Every industry is scaling new heights by tapping into the power of data. Sharpen your skills and become a part of the hottest trend in the 21st century.
"""

# Create a Spacy Doc object from the text
doc = nlp(text)

# Extract features from the text using the nlp object
features = [token.vector for token in doc]

# Get the POS tags for each token using list comprehension
pos_tags = [token.pos_ for token in doc]

# Create a DataFrame
df = pd.DataFrame({'text': [token.text for token in doc], 'grammar': pos_tags})

x_train, x_test, y_train, y_test = train_test_split(features, pos_tags, test_size=0.2, random_state=100)

# Train the SVC model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Make predictions on the training and testing sets
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

# Calculate the accuracy of the predictions
y_train_pred = accuracy_score(y_train, y_train_pred)
y_test_pred = accuracy_score(y_test, y_test_pred)

print(y_train_pred)
print(y_test_pred)

# Speech recognition setup
recognizer = sr.Recognizer()
microphone = sr.Microphone()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)             #female voice
engine.setProperty('rate', 150.5)
engine.setProperty('pitch', 10) 
engine.say("hi there i will do pos_tag work so tell me the sentences")
engine.runAndWait()

while True:
    print("Listening...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    print("Transcribing...")
    try:
        user_input = recognizer.recognize_google(audio)
        print("User input:", user_input)
        engine.say(user_input)                  #generate voice
        engine.runAndWait()
        if(user_input == "exit"):
            print("we are gonna end the program")
            engine.say("we are gonna end the program")    #generate voice
            engine.runAndWait()
            break
        
        user_input_doc = nlp(user_input)
        user_input_features = [token.vector for token in user_input_doc]

        prediction = lr.predict(user_input_features)

        print("Predicted POS tags:", prediction)

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Sorry, an error occurred. Please check your internet connection.")
