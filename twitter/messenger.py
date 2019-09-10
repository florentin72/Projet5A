#Python libraries that we need to import for our bot
import random
from flask import Flask, request
from pymessenger.bot import Bot
import twitterConfig
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import NeuralNetwork as nn



app = Flask(__name__)
ACCESS_TOKEN = twitterConfig.messenger_token
VERIFY_TOKEN = twitterConfig.verification_token
bot = Bot(ACCESS_TOKEN)
data_path = 'data/x.csv'
classification_path = 'data/y.csv'

#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
  

        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
     
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
    
       output = request.get_json()
     
       for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
          
            if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                recipient_id = message['sender']['id']
                if message['message'].get('text'):
                    message_recu = message['message'].get('text')
                  
                    response_sent_text = get_message(message_recu)
                    send_message(recipient_id, response_sent_text)
                #if user sends us a GIF, photo,video, or any other non-text item
                if message['message'].get('attachments'):
                    response_sent_nontext = get_message()
                    send_message(recipient_id, response_sent_nontext)
    return "Message Processed"


def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token you sent
    #if they match, allow the request, else return an error 
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:

            return "Verification token mismatch", 403

        return request.args["hub.challenge"], 200
    return "Hello world", 200




#chooses a random message to send to the user
def get_message(message_recu):
    response = word2Vec(message_recu)
    x = nn.load_samples(data_path)
    y = nn.load_labels(classification_path)

    rnn =  nn.test_with_train_data(x, y)
    res = rnn.predict(x)[0]
    if (str(res) == "0"):
        res = "C'est probablement une fake news !"  
    else:
        res = "C'est probablement vrai ! "
    return  str(res)

#uses PyMessenger to send response to user
def send_message(recipient_id, response):
    #sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"




def word2Vec(message):
    vectorizer = CountVectorizer()
    corpus = [
          'cache derrière Gorafi' , 'humour dénoncer' , 'être gentil Yann Barthès',
          'Paris' ,'Lancement nouveau service livraison domicile mâche repas',
          '@pabl0mira', 'avoir marre répondre.',
          ' @le_gorafi', 'Écrêtage','distraction', 'tétracapillosectomie',
    ]
    X = vectorizer.fit_transform(corpus)# columns of X correspond to the result of this method
    vectorizer.get_feature_names() == (
        ['Gorafi', 'humour', 'avoir', 'distraction', 'tétracapillosectomie',
        'domicile', 'repas', 'dénoncer', 'Lancement', 'text',
        'Écrêtage', 'Paris', 'Barthès'])# retrieving the matrix in the numpy form
    X.toarray()# transforming a new document according to learn vocabulary
    # learn the vocabulary and store CountVectorizer sparse matrix in X
    a = vectorizer.transform([str(message)]).toarray()
    #inputVector.write(vectorizer.transform([token]).toarray())
    np.savetxt("truc.csv", a, delimiter=",",fmt='%d')
    return a 




if __name__ == "__main__":
    app.run(port=8000)