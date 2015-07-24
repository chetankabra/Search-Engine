1. You might have to install the following Python packages: uuid, urlparse, scrapy, pymongo, nltk and networkx. Once you install nltk, run the command  nltk.download() in the IPython terminal and download all corpus related to the book.
2. Install mongodb. Create a database called as uta-edu-corpus. Create an empty table called webpages.
3. PA1: If possible run on Linux or Mac. Scrapy typically runs 50-100 concurrent processes and atomic file operations in Windows are a bit flaky.
4. You can execute the crawler by using the commmand "scrapy runspider pa4_task1_crawler.py"
