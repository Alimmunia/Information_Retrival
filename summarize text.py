import streamlit as st
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import math
from io import StringIO

#Sidebar
sidebar = st.sidebar.radio("MENU", ["Summary Text","About","Our Team"])

if sidebar == "Summary Text":
  #title
  st.title("SUMMARY TEXT APLICATION")
  st.write("Summary Text using the Term Frequency – Inverse Document Frequency (TF-IDF) method")

  #input text
  FileContent = st.text_area("Enter the text you want to summarize in text area !")

  uploaded_file = st.file_uploader("Or choose a file text  that you want to summarize")
  if uploaded_file is not None:
    
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    FileContent = stringio.read()

  #button
  rangkum = st.button("Summarize Now")

  #program
  if rangkum:
    #membuat word frequency
    def create_frequency_table(FileContent) -> dict:
      stopWords = set(stopwords.words("english"))
      words = word_tokenize(FileContent)
      ps = PorterStemmer()

      freqTable = dict()
      for word in words:
        word = ps.stem(word)
        if word in stopWords:
          continue
        if word in freqTable:
          freqTable[word] +=1
        else:
          freqTable[word] =1

      return freqTable

    freqTable = create_frequency_table(FileContent)

    sentences = sent_tokenize(FileContent)

    # term frequency
    def score_sentences(sentences, FreqTable) -> dict:
      sentenceValue = dict()

      for sentence in sentences:
          word_count_in_sentence = (len(word_tokenize(sentence)))
          for wordValue in freqTable:
              if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
                    
          sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

      return sentenceValue

    sentenceValue = score_sentences(sentences, freqTable)

    #pencarian nilai threshold
    def find_average_score(sentenceValue) -> int:
      sumValues = 0
      for entry in sentenceValue:
        sumValues += sentenceValue[entry]

      #average value of a sentence from original text
      average = int(sumValues / len(sentenceValue))

      return average

    # sentenceValue = score_sentences(sentences, freqTable)
    threshold = find_average_score(sentenceValue)

    #pembuatan ringkasan 
    def generate_summary(sentences, sentenceValue, threshold):
      sentence_count = 0
      summary = ''

      for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
              summary += " " + sentence
              sentence_count += 1

      return summary

    summary = generate_summary(sentences, sentenceValue, threshold)

    #Membuat matriks frekuensi dari setiap kata-kata di dalam kalimat
    def create_frequency_matrix(sentences):
      frequency_matrix = {}
      stopWords = set(stopwords.words("english"))
      ps = PorterStemmer()

      for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] +=1
            else:
                freq_table[word] =1
          
        frequency_matrix[sent[:15]] = freq_table
      
      return frequency_matrix

    freq_matrix = create_frequency_matrix(sentences)

    #Hitung TF dan buat matriks untuk menyimpannya
    def create_tf_matrix(freq_matrix):
      tf_matrix = {}

      for sent, f_table in freq_matrix.items():
          tf_table = {}

          count_words_in_sentence = len(f_table)
          for word, count in f_table.items():
              tf_table[word] = count / count_words_in_sentence

          tf_matrix[sent] = tf_table

      return tf_matrix

    tf_matrix = create_tf_matrix(freq_matrix)

    #Membuat table untuk dokumen setiap katanya
    def create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1


        return word_per_doc_table

    count_doc_per_words = create_documents_per_words(freq_matrix)

    #Menghitung nilai IDF dan buat matriks untuk menyimpannya
    def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
          idf_table = {}

          for word in f_table.keys():
              idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

          idf_matrix[sent] = idf_table

        return idf_matrix

    total_documents = len(sentences)
    idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)

    #Menghitung nilai TF-IDF dan buat matriks untuk menyimpannya
    def create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
            
            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):

                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        
        return tf_idf_matrix

    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)

    #Menghitung score dari setiap kalimat
    def score_sentences(idf_matrix) -> dict:
      """
      score a sentences by its word's TF
      basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentences
      :rtype: dict
      """

      sentenceValue = {}

      for sent, f_table in idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
          total_score_per_sentence += score
        
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

      return sentenceValue

    sentenceValue = score_sentences(idf_matrix)

    #average score of all the sentences
    def find_average_score(sentenceValue) -> int:
      """
      find the average score from the sentence value dictionary 
      :rtype: int
      """

      sumValues = 0
      for entry in sentenceValue:
        sumValues += sentenceValue[entry]

      #average value of a sentence from original summary_text

      average = (sumValues / len(sentenceValue))

      return average

    threshold = find_average_score(sentenceValue)

    #
    def generate_summary(sentences, sentenceValue, threshold):
      sentence_count = 0
      summary = ''

      for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
          summary += " " + sentence
          sentence_count += 1

      return summary

    summary_tf_idf = generate_summary(sentences, sentenceValue, threshold)
    st.info(summary_tf_idf)
    st.download_button('Download this text', summary_tf_idf)
    st.balloons()

if sidebar == "About":
  #title
  st.title("ABOUT")
  st.write("This is an application that can apply retrieval information in everyday life. The topic of information retrieval that we chose was the summarization of documents.")
  st.write("Here are the programs that we use in this application:")

  #import libraries
  st.header("Libraries")
  code1 = '''import streamlit as st
  import nltk
  nltk.download('punkt')
  from nltk.tokenize import word_tokenize, sent_tokenize
  from nltk.stem import PorterStemmer
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  import math
  from io import StringIO'''
  st.code(code1, language='python')
  st.write("This code imports several libraries that are used for natural language processing tasks.")
  st.write("- streamlit is a library for creating interactive web applications with Python.")
  st.write("- nltk is the Natural Language Toolkit library, which provides a wide range of tools for working with human language data.")
  st.write("- nltk.download('punkt') downloads the NLTK tokenizer for word and sentence tokenization. The punkt package is a pre-trained unsupervised machine learning model for tokenization.")
  st.write("- from nltk.tokenize import word_tokenize, sent_tokenize imports the word_tokenize and sent_tokenize functions from the nltk.tokenize module. These functions can be used to divide a text into words or sentences, respectively.")
  st.write("- from nltk.stem import PorterStemmer imports the PorterStemmer class from the nltk.stem module. This class can be used to find the root of a word, called a stem, which is useful for text data preprocessing.")
  st.write("- nltk.download('stopwords') downloads a list of commonly used stopwords (such as 'a', 'an', 'the', etc.) that can be filtered out from text data.")
  st.write("- from nltk.corpus import stopwords imports the list of stopwords from the NLTK corpus.")
  st.write("- import math is a library for mathematical operation.")
  st.write("- The io module provides Python’s main facilities for dealing with various types of I/O. The StringIO class is a class from this module that implements a file-like interface for reading or writing strings. It allows you to treat a string as if it were a file, and it can be used as a drop-in replacement for a file object, allowing you to write code that works with files or strings without modification.")
  
  #input text
  st.header("Input text")
  code2 = '''FileContent = st.text_area("Enter the text you want to summarize !")'''
  st.code(code2, language='python')
  st.write('''This code appears to be using the library "streamlit" to create a text input field in which the user can enter a block of text. The text_area() function is creating this input field, and the argument "Enter the text you want to summarize !" is the label that is displayed next to the input field. The input entered by the user is being stored in the variable "FileContent".''')
  code21 = '''uploaded_file = st.file_uploader("Or choose a file text  that you want to summarize")
  if uploaded_file is not None:
    
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    FileContent = stringio.read()'''
  st.code(code21, language='python')
  st.write('''This code is using the Streamlit library to create a file uploader widget in a Python script. The st.file_uploader function creates a button that allows the user to select a file from their computer. The selected file is then stored in the variable uploaded_file.''')
  st.write('''The following if statement checks if a file has been uploaded and if so, it proceeds to read the contents of the file.''')
  st.write('''The bytes_data variable is used to store the contents of the uploaded file in bytes format. The stringio variable is used to convert the bytes data to a string-based IO object.''')
  st.write('''Finally, the FileContent variable is used to store the contents of the file as a string, which can then be used for further processing, such as summarization.''')

  #button
  st.header("Button")
  code3 = '''rangkum = st.button("Summarize Now")'''
  st.code(code3, language='python')
  st.write('''This code is using the library "streamlit" to create a button with the label "Summarize Now" which is stored in the variable "rangkum". When this button is clicked, it will likely trigger some sort of event or function that will use the input text stored in the "FileContent" variable to perform text summarization. The purpose of this button is to allow the user to initiate the summarization process on the text they have entered.''')

  #Word frequency
  st.header("Word frequency")
  code4 = '''if rangkum:
    #membuat word frequency
    def create_frequency_table(FileContent) -> dict:
      stopWords = set(stopwords.words("english"))
      words = word_tokenize(FileContent)
      ps = PorterStemmer()

      freqTable = dict()
      for word in words:
        word = ps.stem(word)
        if word in stopWords:
          continue
        if word in freqTable:
          freqTable[word] +=1
        else:
          freqTable[word] =1

      return freqTable

    freqTable = create_frequency_table(FileContent)

    sentences = sent_tokenize(FileContent)'''
  st.code(code4, language='python')
  st.write('''This code is checking if the variable "rangkum" is true, which likely means that the "Summarize Now" button has been clicked. If the button has been clicked, the code defines a function create_frequency_table() which takes in a variable "FileContent" as its parameter.''')
  st.write('''This function tokenizes the words in the text and creates a frequency table of all the words in the text, with the number of occurrences of each word. The function uses the Natural Language Toolkit (NLTK) library to tokenize the words and remove stop words, which are common words such as "a", "an", "the", etc. that do not add much meaning to the text. The function also applies stemming, which is the process of reducing inflected or derived words to their word stem, base or root form.''')
  st.write('''It then stores the frequency table in the variable freqTable and tokenize the sentences using sent_tokenize function and store it in sentences. This frequency table and sentences will likely be used later in the code to perform text summarization.''')

  #term frequency
  st.header("Term frequency")
  code5 = '''def score_sentences(sentences, FreqTable) -> dict:
      sentenceValue = dict()

      for sentence in sentences:
          word_count_in_sentence = (len(word_tokenize(sentence)))
          for wordValue in freqTable:
              if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
                    
          sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

      return sentenceValue

    sentenceValue = score_sentences(sentences, freqTable)'''
  st.code(code5, language='python')
  st.write('''This code defines a function score_sentences() which takes in two arguments: sentences and FreqTable. The purpose of this function is to assign a score to each sentence in the input text, based on the words it contains. The function first initializes an empty dictionary called sentenceValue.''')
  st.write('''Then, the function iterates over each sentence in the sentences list and tokenize the words in the sentence. It then counts the number of words in that sentence and stores it in word_count_in_sentence. Next, it iterates over each word in the FreqTable and check if the word is present in the sentence. If the word is present in the sentence, the function adds the word's frequency to the score for that sentence. If the sentence is not yet in the sentenceValue dictionary, it is added and the word's frequency is added to the score.''')
  st.write('''Finally, the function normalizes the sentence score by dividing it by the number of words in the sentence and stores the result in the sentenceValue dictionary. The function returns the sentenceValue dictionary, which will likely be used later in the code to select the sentences with the highest scores as the summary of the text.''')
  st.write('''After that, the score_sentences function is called and the results are stored in the sentenceValue variable. This variable will likely be used later in the code to extract the most informative sentences.''')

  #Threshold
  st.header("Threshold")
  code6 = '''def find_average_score(sentenceValue) -> int:
      sumValues = 0
      for entry in sentenceValue:
        sumValues += sentenceValue[entry]

      #average value of a sentence from original text
      average = int(sumValues / len(sentenceValue))

      return average

    # sentenceValue = score_sentences(sentences, freqTable)
    threshold = find_average_score(sentenceValue)'''
  st.code(code6, language='python')
  st.write('''This code defines a function find_average_score() which takes in one argument sentenceValue, which is a dictionary that contains the scores for each sentence in the input text, as determined by the score_sentences() function.''')
  st.write('''The function first initializes a variable sumValues to zero, then iterates over each entry in the sentenceValue dictionary and adds the score for that sentence to sumValues. It then calculates the average score by dividing sumValues by the number of sentences and converts the result to an integer. The function returns this average score.''')
  st.write('''After that, the function is called and the results are stored in the threshold variable. This threshold variable will likely be used later in the code as a cutoff point for selecting the sentences that will be included in the summary. Sentences with scores above this threshold will be considered as the summary.''')

  #provisional summary
  st.header("Provisional summary")
  code7 = '''def generate_summary(sentences, sentenceValue, threshold):
      sentence_count = 0
      summary = ''

      for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
              summary += " " + sentence
              sentence_count += 1

      return summary

    summary = generate_summary(sentences, sentenceValue, threshold)'''
  st.code(code7, language='python')
  st.write('''This code defines a function generate_summary() which takes in three arguments: sentences, sentenceValue and threshold. The function initializes a variable sentence_count to zero and summary to an empty string. Then, it iterates over each sentence in the sentences list.''')
  st.write('''For each sentence, the function checks if the sentence's first 10 characters are in the sentenceValue dictionary and if the score for that sentence is greater than the threshold. If both conditions are true, the function adds the sentence to the summary string, and increments sentence_count.''')
  st.write('''At the end of the loop, the function returns the summary which contains the selected sentences that have scores greater than the threshold.''')
  st.write('''After that, the generate_summary function is called, and the results are stored in the summary variable. This variable will likely be used later in the code to display the summary to the user.''')

  #frequency_matrix
  st.header("Frequency matrix")
  code8 = '''def create_frequency_matrix(sentences):
      frequency_matrix = {}
      stopWords = set(stopwords.words("english"))
      ps = PorterStemmer()

      for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] +=1
            else:
                freq_table[word] =1
          
        frequency_matrix[sent[:15]] = freq_table
      
      return frequency_matrix

    freq_matrix = create_frequency_matrix(sentences)'''
  st.code(code8, language='python')
  st.write('''This code defines a function create_frequency_matrix() which takes in one argument: sentences. The function's purpose is to create a frequency matrix of all the words in the input text.''')
  st.write('''The function initializes an empty dictionary called frequency_matrix and creates a set of stop words using the NLTK library. The function then iterates over each sentence in the sentences list, and for each sentence, it creates a frequency table of all the words in that sentence, with the number of occurrences of each word. The function tokenizes the words in the sentence, removes stop words and applies stemming.''')
  st.write('''It then stores the frequency table in the frequency_matrix dictionary, using the first 15 characters of the sentence as the key. The function returns the frequency_matrix which will likely be used later in the code for further processing or analysis.''')
  st.write('''After that, the create_frequency_matrix function is called and the results are stored in the freq_matrix variable. This variable will likely be used later in the code to calculate the importance of each sentence in the text.''')

  #tf_matrix
  st.header("TF Matrix")
  code9 = '''def create_tf_matrix(freq_matrix):
      tf_matrix = {}

      for sent, f_table in freq_matrix.items():
          tf_table = {}

          count_words_in_sentence = len(f_table)
          for word, count in f_table.items():
              tf_table[word] = count / count_words_in_sentence

          tf_matrix[sent] = tf_table

      return tf_matrix

    tf_matrix = create_tf_matrix(freq_matrix)'''
  st.code(code9, language='python')
  st.write('''This code defines a function create_tf_matrix() which takes in one argument freq_matrix. The function's purpose is to create a Term Frequency (TF) matrix from the frequency matrix of all the words in the input text.''')
  st.write('''The function initializes an empty dictionary called tf_matrix. Then, it iterates over each sentence in the freq_matrix, using the sentence as the key, and the frequency table of words in that sentence as the value.''')
  st.write('''For each sentence, the function creates a new table called tf_table, which will contain the TF values for each word in that sentence. It starts by counting the total number of words in the sentence, and then for each word in the frequency table, it calculates the TF value for that word by dividing the word's frequency by the total number of words in the sentence.''')
  st.write('''The function then adds the tf_table to the tf_matrix dictionary using the sentence as the key. The function returns the tf_matrix which will likely be used later in the code for further processing or analysis.''')
  st.write('''After that, the create_tf_matrix function is called and the results are stored in the tf_matrix variable. This variable will likely be used later in the code to calculate the importance of each word in the text.''')


  #documents_per_words
  st.header("Documents per Words")
  code10 = '''def create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1


        return word_per_doc_table

    count_doc_per_words = create_documents_per_words(freq_matrix)'''
  st.code(code10, language='python')
  st.write('''This code defines a function create_documents_per_words() which takes in one argument freq_matrix. The function's purpose is to create a table which contains the count of the number of documents in which each word appears.''')
  st.write('''The function initializes an empty dictionary called word_per_doc_table. Then, it iterates over each sentence in the freq_matrix, using the sentence as the key, and the frequency table of words in that sentence as the value.''')
  st.write('''For each sentence, it iterates over each word in the frequency table, and if the word is already in the word_per_doc_table dictionary, it increments the count by 1. If the word is not in the word_per_doc_table dictionary, it adds it with the count of 1.''')
  st.write('''The function returns the word_per_doc_table which will likely be used later in the code for further processing or analysis.''')
  st.write('''After that, the create_documents_per_words function is called and the results are stored in the count_doc_per_words variable. This variable will likely be used later in the code to calculate the Inverse Document Frequency (IDF) of each word in the text.''')

  #idf_matrix
  st.header("IDF Matrix")
  code11 = '''def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
          idf_table = {}

          for word in f_table.keys():
              idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

          idf_matrix[sent] = idf_table

        return idf_matrix

    total_documents = len(sentences)
    idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)'''
  st.code(code11, language='python')
  st.write('''This code defines a function create_idf_matrix() which takes in three arguments: freq_matrix, count_doc_per_words, and total_documents. The function's purpose is to create an Inverse Document Frequency (IDF) matrix from the frequency matrix of all the words in the input text and the count of documents in which each word appears.''')
  st.write('''The function initializes an empty dictionary called idf_matrix. Then, it iterates over each sentence in the freq_matrix, using the sentence as the key, and the frequency table of words in that sentence as the value.''')
  st.write('''For each sentence, the function creates a new table called idf_table, which will contain the IDF values for each word in that sentence. It starts by calculating the IDF values of each word by taking the logarithm base 10 of the ratio between the total number of documents and the number of documents in which the word appears.''')
  st.write('''The function then adds the idf_table to the idf_matrix dictionary using the sentence as the key. The function returns the idf_matrix which will likely be used later in the code for further processing or analysis.''')
  st.write('''After that, the total_documents variable is defined as the length of the sentences in the input text and the create_idf_matrix function is called with the freq_matrix, count_doc_per_words and total_documents variables as arguments, and the results are stored in the idf_matrix variable''')

  #tf_idf_matrix
  st.header("TF IDF Matrix")
  code12 = '''def create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
            
            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):

                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        
        return tf_idf_matrix

    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)'''
  st.code(code12, language='python')
  st.write('''This code defines a function create_tf_idf_matrix() which takes in two arguments: tf_matrix and idf_matrix. The function's purpose is to create a TF-IDF matrix, which is a numerical representation of the importance of each word in the input text.''')
  st.write('''The function initializes an empty dictionary called tf_idf_matrix. Then, it iterates over each sentence in the tf_matrix and idf_matrix, using the sentence as the key, and the frequency table of words in that sentence as the value.''')
  st.write('''For each sentence, the function creates a new table called tf_idf_table, which will contain the TF-IDF values for each word in that sentence. It starts by calculating the TF-IDF values of each word by multiplying the TF and IDF values of that word obtained from tf_matrix and idf_matrix respectively.''')
  st.write('''The function then adds the tf_idf_table to the tf_idf_matrix dictionary using the sentence as the key. The function returns the tf_idf_matrix which will likely be used later in the code for further processing or analysis.''')
  st.write('''After that, the create_tf_idf_matrix function is called with the tf_matrix and idf_matrix variables as arguments, and the results are stored in the `''')

  #score_sentences
  st.header("Score sentences")
  code13 = '''def score_sentences(idf_matrix) -> dict:
      """
      score a sentences by its word's TF
      basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentences
      :rtype: dict
      """

      sentenceValue = {}

      for sent, f_table in idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
          total_score_per_sentence += score
        
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

      return sentenceValue

    sentenceValue = score_sentences(idf_matrix)'''
  st.code(code13, language='python')
  st.write('''This code defines a function score_sentences() which takes in one argument: idf_matrix. The function's purpose is to score each sentence in the input text based on the TF-IDF values of the words in that sentence.''')
  st.write('''The function initializes an empty dictionary called sentenceValue. Then, it iterates over each sentence in the idf_matrix, using the sentence as the key, and the frequency table of words in that sentence as the value.''')
  st.write('''For each sentence, the function starts by initializing a variable total_score_per_sentence to zero, it then counts the total number of words in the sentence, then for each word in the frequency table, it adds the TF-IDF score for that word to total_score_per_sentence.''')
  st.write('''The function then calculates the average score for the sentence by dividing the total_score_per_sentence by the number of words in the sentence and stores it in the sentenceValue dictionary using the sentence as the key.''')
  st.write('''The function returns the sentenceValue which will likely be used later in the code as a measure of the importance of each sentence in the text.''')

  #average score of all the sentences
  st.header("Average score of all the sentences")
  code14 = '''def find_average_score(sentenceValue) -> int:
      """
      find the average score from the sentence value dictionary 
      :rtype: int
      """

      sumValues = 0
      for entry in sentenceValue:
        sumValues += sentenceValue[entry]

      #average value of a sentence from original summary_text

      average = (sumValues / len(sentenceValue))

      return average

    threshold = find_average_score(sentenceValue)'''
  st.code(code14, language='python')
  st.write('''This code defines a function find_average_score() which takes in one argument: sentenceValue. The function's purpose is to find the average score of all the sentences in the input text.''')
  st.write('''The function starts by initializing a variable sumValues to zero. Then, it iterates over each sentence in the sentenceValue dictionary and adds the score for that sentence to the sumValues variable.''')
  st.write('''After iterating over all sentences, it calculates the average score by dividing the sumValues by the number of sentences.''')
  st.write('''The function returns the average score which will likely be used later in the code as a threshold to determine which sentences should be included in the summary.''')

  #generate_summary
  st.header("Generate summary")
  code15 = '''def generate_summary(sentences, sentenceValue, threshold):
      sentence_count = 0
      summary = ''

      for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
          summary += " " + sentence
          sentence_count += 1

      return summary

    summary_tf_idf = generate_summary(sentences, sentenceValue, threshold)'''
  st.code(code15, language='python')
  st.write('''This code defines a function generate_summary() which takes in three arguments: sentences, sentenceValue, and threshold. The function's purpose is to generate a summary of the input text by selecting the sentences with scores greater than or equal to the threshold value.''')
  st.write('''The function starts by initializing a variable sentence_count to zero, and an empty string summary. Then, it iterates over each sentence in the sentences list. For each sentence, it checks if that sentence exists in the sentenceValue dictionary and if its score is greater than or equal to the threshold value. If the condition is true, it adds the sentence to the summary string and increments the sentence_count variable by 1.''')
  st.write('''The function returns the summary string which is a summary of the original text. After that the generate_summary function is called with the sentences, sentenceValue, and threshold variables as arguments, and the result is stored in the summary_tf_idf variable.''')

  #results and download
  st.header("Results and Download")
  code6 = '''st.info(summary_tf_idf)
  st.download_button('Download this text', summary_tf_idf)'''
  st.code(code6, language='python')
  st.write('''The first line st.info(summary_tf_idf) is using the streamlit.info() function to display the summary_tf_idf on the screen. This function displays a message in a non-interactive element, usually used to show some information to the user.''')
  st.write('''The second line st.download_button('Download this text', summary_tf_idf) is using the streamlit.download_button() function to create a button that when clicked, downloads the summary_tf_idf as a text file. This function takes two arguments: the first is the text that will be displayed on the button, and the second is the variable or the text that will be downloaded.''')
  st.write('''In summary, the code is displaying the summary on the screen and providing a button for the user to download the summary as a text file.''')

if sidebar == "Our Team":
  #title
  st.title("OUR TEAM :")
  st.write("1. 0110221227 - Huzaifah Alim -> (Leader)")
  st.write("2. 0110221224 - Fauziyyah Annisah")
  st.write("3. 0110221216 - Haniefa Aulia Rahma")

