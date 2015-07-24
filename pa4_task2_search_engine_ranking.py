#####begin put all your imports
import nltk

import pymongo

from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import re
import numpy
import string
import math
from nltk.corpus import stopwords
#####end put all your imports


#mongodb_client = None
#mongodb_db = None
mongodb_client = pymongo.MongoClient()
mongodb_db = mongodb_client['uta-edu-corpus']
document_frequency = defaultdict(int)
total_number_of_docs = 0

def setup_mongodb():
    #####################Task t2a: your code #######################
    #Connect to mongodb
    mongodb_client = pymongo.MongoClient()
    mongodb_db = mongodb_client['uta-edu-corpus']
    #####################Task t2a: your code #######################

#This function processes the entire document corpus
def process_document_corpus(file_name):
    #####################Task t2b: your code #######################
    #### The input is a file where in each line you have two information
    #   filename and url separated by |
    # Process the file line by line
    #   and for each line call the function process_document with file name and url and index
    #   first file should have an index of 0, second has 1 and so on
    #Remember to set total_number_of_docs to number of documents
    f = open(file_name)
    lines_in_file = []
    print 'stopwords::',stopwords.words('english')
    for line in f:
        #word_list = line.split('|')
        #process_document(word_list[0],word_list[1],total_number_of_docs)
        lines_in_file.append(line)
    total_number_of_docs = len(lines_in_file)
    globals()['total_number_of_docs'] = total_number_of_docs
    print total_number_of_docs
    #rohan = []

    for i in range(0, total_number_of_docs):
        data = lines_in_file[i]
        split_data = data.split('|')
        process_document(split_data[0], split_data[1], i)
        #break
    #print rohan
    #####################Task t2b: your code #######################
    pass


#This function processes a single web page and inserts it into mongodb
def process_document(file_name, url, index):

    #Do not change
    f = open(file_name)
    file_contents = f.read()
    f.close()

    soup = BeautifulSoup(file_contents)


    #####################Task t2c: your code #######################
    #Using the functions that you will write (below), convert the document
    #   into the following structure:
    # title_processed: a string that contains the title of the string after processing
    # hx_processed: an array of strings where each element is a processed string
    #   for eg, if the document has two h1 tags, then the array h1_processed will have two elements
    #   one for each h1 tag and contains its contentent after processing
    # a_processed: same for a tags
    # body_processed: a string that contains body of the document after processing
    #print 'title' + soup.title.string

    title_processed = soup.title.string

    h1_processed = []
    h1_tag = soup.find_all('h1')
    h1_text = []
    for i in range(0,len(h1_tag)):
        h1_text.append(h1_tag[i].get_text())
    h1_processed = process_array(h1_text)
    #print h1_processed

    h2_processed = []
    h2_tag = soup.find_all('h2')
    h2_text = []
    for i in range(0,len(h2_tag)):
        h2_text.append(h2_tag[i].get_text())
    h2_processed = process_array(h2_text)
    #print h2_processed

    h3_processed = []
    h3_tag = soup.find_all('h3')
    h3_text = []
    for i in range(0,len(h3_tag)):
        h3_text.append(h3_tag[i].get_text())
    h3_processed = process_array(h3_text)
    #print h3_processed

    h4_processed = []
    h4_tag = soup.find_all('h4')
    h4_text = []
    for i in range(0,len(h4_tag)):
        h4_text.append(h4_tag[i].get_text())
    h4_processed = process_array(h4_text)
    #print h4_processed

    h5_processed = []
    h5_tag = soup.find_all('h5')
    h5_text = []
    for i in range(0,len(h5_tag)):
        h5_text.append(h5_tag[i].get_text())
    h5_processed = process_array(h5_text)
    #print h5_processed

    h6_processed = []
    h6_tag = soup.find_all('h6')
    h6_text = []
    for i in range(0,len(h6_tag)):
        h6_text.append(h6_tag[i].get_text())
    h6_processed = process_array(h6_text)
    #print h6_processed

    a_processed = []
    a_tag = soup.find_all('a')
    a_text = []
    for i in range(0,len(a_tag)):
        a_text.append(a_tag[i].get_text())
    a_processed = process_array(a_text)
    #print a_processed

    body_content = soup.find('body')
    body_text = []
    body_text.append(body_content.get_text())
    body_processed = process_array(body_text)
    #print body_processed
    #Insert the processed document into mongodb
    #Do not change
    webpages = mongodb_db.webpages
    document_to_insert = {
        "url": url,
        "title": title_processed,
        "h1": h1_processed,
        "h2": h2_processed,
        "h3": h3_processed,
        "h4": h4_processed,
        "h5": h5_processed,
        "h6": h6_processed,
        "a" : a_processed,
        "body": body_processed,
        "filename": file_name,
        "index": index
        }

    webpage_id = webpages.insert_one(document_to_insert)
    body_processed = ' '.join(body_processed).encode('utf-8')
    #####################Task t2c: your code #######################

    #Do not change below
    #Write the processed document
    new_file_name = file_name.replace("downloads/", "processed/")
    f = open("processedFileNamesToUUID.txt", "a")
    f.write(new_file_name + "|" + url)
    f.flush()
    f.close()

    f = open(new_file_name, "w")
    f.write(body_processed)
    f.close()



#helper function for h tags and a tags
# use if needed
def process_array(array):
    processed_array = [process_text(element) for element in array]
    return processed_array

#This function does the necessary text processing of the text
def process_text(text):
    processed_text = ""

    #####################Task t2d: your code #######################
    #Given the text, do the following:
    #   convert it to lower case
    #   remove all stop words (English)
    #   remove all punctuation
    #   stem them using Porter Stemmer
    #   Lemmatize it
    lower_text = text.lower()

    stopwords = sorted(nltk.corpus.stopwords.words('english'))

    tokenized_text = sorted(nltk.word_tokenize(lower_text))
    stopword_filtered_list = []
    for word in tokenized_text:
        if word not in stopwords:
            stopword_filtered_list.append(word)


    #######################
    exclude = set(string.punctuation)
    table = string.maketrans("","")
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    for i in range(0,len(stopword_filtered_list)):
        #word.translate(table, string.punctuation)
        word = stopword_filtered_list[i]
        stopword_filtered_list[i] = regex.sub('', word)
    ########################

    #for word in tokenized_text:
     #   if not word.isalpha():
      #      tokenized_text.remove(word)
    stemmer = PorterStemmer()
    stemmed_list = []
    for word in stopword_filtered_list:
        stemmed_list.append(stemmer.stem(word))


    lemmetized_list = []
    lemmetizor = WordNetLemmatizer()
    for word in stemmed_list:
        lemmetized_list.append(lemmetizor.lemmatize(word))
    #print lemmetized_list
    processed_text = ' '.join(lemmetized_list)

    #####################Task t2d: your code #######################

    return processed_text


#This function determines the vocabulary after processing
def find_corpus_vocabulary(file_name):
    vocabulary = None
    top_5000_words = None
    #Document frequency is a dictionary
    #   given a word, it will tell you how many documents this word was present it
    # use the variable document_frequency
    document_frequency = defaultdict(int)

    #####################Task t2e: your code #######################
    # The input is the file name with url and processed file names
    # for each file:
    #   get all the words and compute its frequency (over the entire corpus)
    # return the 5000 words with highest frequency
    # Hint: check out Counter class in Python
    file_reader = open(file_name,mode='r')
    counter = Counter()
    for line in file_reader:
        file = line.split("|")[0]
        reader = open(file,mode='r')
        tokenized_file = nltk.word_tokenize(reader.read())
        counter.update(tokenized_file)
        for key in counter:
           if key in tokenized_file:
                document_frequency[key] = document_frequency.get(key,0)+1
    globals()['document_frequency'] = document_frequency
    #print document_frequency
    #print globals()['document_frequency']
    most_common = counter.most_common(5000)

    top_5000_words = []
    for tuple in most_common:
        top_5000_words.append(tuple[0])
    #print top_5000_words
    #####################Task t2e: your code #######################

    f = open("vocabulary.txt", "w")
    for word in top_5000_words:
        f.write(word + "," + str(document_frequency[word]) + "\n")
    f.close()

    return top_5000_words


def corpus_to_document_vec(vocabulary_file_name, file_name, output_file_name):
    #####################Task t2f: your code #######################
    # The input is the file names of vocabulary, and
    #   the file  with url, processed file names  and the output file name
    #   the output is a file with tf-idf vector for each document
    #Pseudocode:
    # for each file:
    #   call the function text_to_vec with document body
    #   write the vector into output_file_name one line at a time
    #   into output_file_name
    #       ie document i will be in the i-th line
    vocabulary_text = open(vocabulary_file_name).read()
    file_name_list = open(file_name)
    output_file = open(output_file_name,'w')
    i = 0
    for line in file_name_list:
        processed_file_name = line.split('|')[0]
        processed_file_content = open(processed_file_name).read()
        #print vocabulary_text
        #print processed_file_content
        document_vector = text_to_vec(vocabulary_text,processed_file_content)
        #print document_vector
        for element in document_vector:
            output_file.write(str(element) + ',')
        output_file.write('\n')

    output_file.close()
    #####################Task t2f: your code #######################
    pass


def text_to_vec(vocabulary, text):
    #####################Task t2g: your code #######################
    # The input are vocabulary and text
    #   compute its tf-idf vector (ignore all words not in vocabulary)
    #Remember to use the variable document_frequency for computing idf

    #print document_frequency
    tokenize_text = nltk.word_tokenize(text)
    #print tokenize_text
    tf_idf_vector = numpy.zeros(5000)
    i = 0
    total_rohan = 0
    for line in vocabulary.splitlines():
        word = line.split(',')[0]
        count = tokenize_text.count(word)
        word_freq_document = globals()['document_frequency'].get(word)
        #print 'word  ' , word
        #print 'count ' , count
        #print 'freq ' , word_freq_document
        if count == 0:
            tf = 0
        else:
            tf = 1+math.log(count)
        dem = float(total_number_of_docs)/float(word_freq_document)
        idf = math.log(dem)
        tf_idf_vector[i] = tf * idf
        #print 'count' , count
        #print 'word ' , word
        #print 'freq ' , word_freq_document
        i+=1
        #print tf
    tf_idf_vector = tf_idf_vector/numpy.linalg.norm(tf_idf_vector)
    print numpy.linalg.norm(tf_idf_vector)
    return tf_idf_vector
    #####################Task t2g: your code #######################
    pass


def query_document_similarity(query_vec, document_vec):
    #####################Task t2h: your code #######################
    #   Given a query and document vector
    #   compute their cosine similarity
    numerator = 0
    query_vec_square_sum = 0
    document_vec_square_sum = 0
    for i in range(0,len(document_vec)):
        numerator=numerator+(query_vec[i]*document_vec[i])
        query_vec_square_sum+=query_vec[i]*query_vec[i]
        document_vec_square_sum+=document_vec_square_sum[i]*document_vec_square_sum[i]
    print numerator
    print query_vec_square_sum
    print document_vec_square_sum
    denominator = math.sqrt(query_vec_square_sum)*math.sqrt(document_vec_square_sum)

    cosine_similarity = numerator/denominator
    #####################Task t2h: your code #######################
    return cosine_similarity

def rank_documents_tf_idf(query, k=10):
    #####################Task t2i: your code #######################

    #convert query to document using text_to_vec function
    query_as_document = None
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   for each matching document:
    #       retrieve its tf-idf vector (use the file_name and index fields from mongodb)
    #   compute the tf-idf score and sort them accordingly
    # return top-k documents


    proceessed_query = process_text(query)
    #mongodb_db.webpages.find({"h1" : /.*query.*/)
    #####################Task t2i: your code #######################
    return ranked_documents[:k]


def rank_documents_zone_scoring(query, k=10):
    #####################Task t2j: your code #######################

    #convert query to document using text_to_vec function
    query_as_document = None
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   for each matching document compute its score as following:
    #       score = 0
    #       for each word in query:
    #           find which "zone" the word fell in and give appropriate score
    #           title = 0.3, h1 = 0.2, h2=0.1, h3=h4=h5=h6=0.05,a: 0.1, body: 0.1
    #   so if a query keyword occured in title, h1 and body, its score is 0.6
    #       compute this score for all keywords
    #       score of the document is the score of all keywords
    # return top-k documents
    #####################Task t2j: your code #######################
    return ranked_documents[:k]

def rank_documents_pagerank(query, k=10):
    #####################Task t2k: your code #######################

    #convert query to document using text_to_vec function
    query_as_document = None
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   order the documents based on their pagerank score (computed in task 3)
    # return top-k documents
    #####################Task t2k: your code #######################
    return ranked_documents[:k]


#Do not change below
def rank_documents(query):
    print "Ranking documents for query:", query
    print "Top-k for TF-IDF"
    print rank_documents_tf_idf(query)
    print "Top-k for Zone Score"
    print rank_documents_zone_scoring(query)
    print "Top-k for Page Rank"
    print rank_documents_pagerank(query)

def test():
    rrrr = ['rohan','rohan','shimpi','test','sdfasdf','rohan']
    print rrrr.count('rohan')

    t = numpy.zeros(5)
    for i in range(0,len(t)):
        t[i]=i
    print t
    print numpy.linalg.norm(t)
    arr =  t/numpy.linalg.norm(t)
    print arr
    print numpy.linalg.norm(arr)
    t= numpy.linalg.norm(t/numpy.linalg.norm(t))
    #t = t/numpy.linalg.norm(t)
    print t

#setup_mongodb()
#####Uncomment the following functions as needed
process_document_corpus("fileNamesToUUID.txt")
#process_text("rohan me shimpi him cats cat reading read glass glasses are is abaci")
vocabulary = find_corpus_vocabulary("processedFileNamesToUUID.txt")
corpus_to_document_vec("vocabulary.txt", "processedFileNamesToUUID.txt", "tf_idf_vector.txt")
#test()