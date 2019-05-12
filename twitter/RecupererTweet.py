import twitter
import twitterConfig
import sys

print(sys.argv[1] )

sourceTweet = sys.argv[1] 
fichierDestination = sys.argv[2]

consumer_key = twitterConfig.consumer_key
consumer_secret = twitterConfig.consumer_secret
access_token_key = twitterConfig.access_token_key
access_token_secret =  twitterConfig.access_token_secret

api = twitter.Api(consumer_key= consumer_key,
  consumer_secret= consumer_secret,
  access_token_key= access_token_key,
  access_token_secret= access_token_secret)



f = open(fichierDestination,'w')

search = api.GetSearch(sourceTweet) 
for tweet in search:
    f.write(tweet.text + '\n')

f.close()
