class TweetCrawler:
    def __init__(self, auth_token, search_keyword, limit, start_date, end_date):
        self.auth_token = auth_token
        self.search_keyword = search_keyword
        self.limit = limit
        self.start_date = start_date
        self.end_date = end_date

    def harvest_tweets(self):
        pass
