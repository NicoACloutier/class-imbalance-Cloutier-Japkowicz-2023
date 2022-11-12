import snscrape.modules.twitter as sntwitter
import pandas as pd

LIMIT = 1000

#transpose a matrix
def transpose(matrix):
    num_rows = len(matrix[0])
    num_columns = len(matrix)
    Final = [[0]*num_columns for x in range(num_rows)]
    for x in range(num_rows):
        for i in range(num_columns):
            Final[x][i] = matrix[i][x]
    return Final
    
#change an int to a string. if it's only made up of one character,
#add a zero to the beginning. (this is because of the way twitter 
#handles dates)
def add_zero(date):
    date = str(date)
    if len(date) == 1:
        date = '0' + date
    return date

def main():
    
    start_dates = [x+1 for x in range(11)]
    
    tweets = []
    for date_index, start_date in enumerate(start_dates):
        end_date = start_date + 1
        start_date = add_zero(start_date) #convert to str and add 0 to start if one digit
        end_date = add_zero(end_date)
        query = f" lang:en until:2021-{end_date}-01 since:2021-{start_date}-01" #the query submitted to twitter
        for index, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            tweets.append([date_index+1, tweet.content]) #appends each tweet we get to the list with the month it's from
            if index == LIMIT-1: #stop once the limit has been reached
                break
    tweets = transpose(tweets)
    tweet_df = pd.DataFrame(tweets, ['Month', 'text'])
    tweet_df = tweet_df.transpose()
    tweet_df.to_csv('..\\data\\normal_tweets.csv', index=False)

if __name__ == '__main__':
    main() 
