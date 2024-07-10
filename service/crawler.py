import subprocess

class TweetCrawler:
    def __init__(self, auth_token, search_keyword, limit, start_date, end_date):
        self.auth_token = auth_token
        self.search_keyword = search_keyword
        self.limit = limit
        self.start_date = start_date
        self.end_date = end_date
        self.filename = search_keyword.split(' ')[0]


    def harvest_tweets(self):
        command = f'npx tweet-harvest -o "{self.filename}" -s "{self.search_keyword} lang:id until:{self.end_date} since:{self.start_date}" -l {self.limit} --token {self.auth_token}'
        print(command)
        subprocess.run(command, shell=True)