from autocorrect import Speller

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class ChatBot(object):

    def __init__(self, data_path='intent_data/'):

        self.cat_list = ['email', 'greeting', 'product_questions', 'store_time']

        self.dataset = load_files(data_path, categories=self.cat_list, encoding='utf-8', random_state=17)

        self.spell = Speller()

        self.intent_clf = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('transformer', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])

        self.intent_clf.fit(self.dataset.data, self.dataset.target)
        
        self.ask_for_email = False
        self.ask_count = 0

    def __call__(self, user_input):

        preprocessed_input = [self.spell(user_input)]
        predicted_intent = self.intent_clf.predict(preprocessed_input)

        if predicted_intent == 0:
            print("All good!")
            self.ask_for_email = True
            return

        if predicted_intent == 1:
            print("Hello there!")
            return
        
        if predicted_intent == 2:
            print("Lets see... we sell fresh fruits, vegies, dairy products and dietary suppliments")
            self.ask_email(user_input)
            return
            
        if predicted_intent == 3:
            print("We are open Monday to Friday from 9:00 AM to 9:00 PM except on holidays")
            self.ask_email(user_input)
            return

        else:
            print("Sorry, I couldn't understand that")
            return
        
    def ask_email(self, user_input):

        if self.ask_for_email is False:

            if self.ask_count == 0:
                print("If you would like to stay up to date, please consider signing up")
                self.ask_count += 1
                return

            elif self.ask_count == 1:
                print("So where was I? Oh yes, please consider signing up to stay up to date")
                self.ask_count += 1
                return

            elif self.ask_count == 2:
                self.ask_for_email = True
            
bot = ChatBot()

if __name__ == '__main__':
    while True:
        x = input(">")
        bot(x)