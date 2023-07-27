import imp
import re
import numpy as np
from random import shuffle
import random
import langid
import pandas as pd

from data_processing.data_utils import *



def printMessage(line, code):
    """for debug"""
    if code == 0:
        print("+++++: {}".format(line))
    else:
        print("--{}--: {}".format(code, line))

def is_english(sentence):
    '''
    Identify whether the sentence is in English
    '''
    if langid.classify(sentence)[0] == 'en':
        return True
    else:
        return False


def checkSentencesLen(line):
    """
    Filter lines containing sentences with fewer than 5 words
    """
    sentences = line.split('.')
    if len(sentences) == 1:
        return False
    for sentence in sentences:
        if len(sentence) == 0 or sentence == ' ' or ' ' not in sentence:
            continue
        if len(sentence.split()) < 5:
            return False
    return True


def checkParenthesesMatch(line):
    """Filter rows with mismatched parentheses ()"""
    stackLen = 0
    for ch in line:
        if ch == '(':
            stackLen += 1
        elif ch == ')':
            stackLen -= 1
            if stackLen < 0:
                return False
    return stackLen == 0


def checkUpperWords(line):
    """Filter lines with two consecutive words in all uppercase"""

    def allUpper(word):
        for ch in word:
            if not 'A' <= ch <= 'Z':
                return False
        return True

    cntUpperWord = 0
    for word in line.split():
        if allUpper(word):
            cntUpperWord += 1
            if cntUpperWord == 2:
                return False
        else:
            cntUpperWord = 0
    return True


def checkMeanWordLength(line):
    """Filter lines with average word length less than 3"""
    totalLen = 0
    totalCnt = 0
    for word in line.split():
        totalLen += len(word)
        totalCnt += 1
    return totalLen / totalCnt > 3


def process_raw_corpus(raw_corpus_path,raw_corpus_filter_path):
    '''
    process the raw corpus in raw_corpus_path and save in raw_corpus_filter_path
    '''
    lines = []
    p1 = re.compile(r'(,|\?|!|;|:|\'|"|\)|\(|\[|\]|\}|\{|<=|>=|=|<|>|-|\+|\*|\/)')
    p2 = re.compile(r'\s+')
    p3 = re.compile(r" ([^\.]+?)\. ")

    radio_corpus = read_txt(raw_corpus_path)
    print('total {num} lines befor filter'.format(num = len(radio_corpus)))

    filter_lines = []


    for line in radio_corpus:
        # Restore the word with line break, for example, restore "hel lo" to "hello"
        line = line.replace('- ', '')
        # Add spaces before and after various symbols to separate them from words, for example, change "Hello,world" to "Hello, world"
        line = p1.sub(r' \1 ', line)
        line = line.replace("' s", "'s")
        # Change multiple space characters into one space
        line = p2.sub(' ', line)

        # Filter lines with length less than 30 words
        if len(line.split()) < 30:
            # printMessage(line, 1)
            continue
        # Filter lines with a length greater than 350 words, because the input of Bert needs to be less than 512 tokens
        if len(line.split()) > 350:
            # printMessage(line, 2)
            continue
        if not checkMeanWordLength(line):
            # printMessage(line, 3)
            continue
        if not checkSentencesLen(line):
            # printMessage(line, 4)
            continue
        if not checkParenthesesMatch(line):
            # printMessage(line, 5)
            continue
        if not checkUpperWords(line):
            # printMessage(line, 6)
            continue

        # Make the text all lowercase to avoid the same word being considered as different words due to different case
        line = line.lower()
        # Separate periods from words with spaces
        line = p3.sub(r" \1 . ", line)

        # printMessage(line, 0)
        if line.endswith('\n') == False:
            line = line + '\n'
        
        if is_english(line):
            filter_lines.append(line)
    
    # random.seed(2022)
    # shuffle(filter_lines)

    print('total {num} lines after filter'.format(num = len(filter_lines)))
    #save
    write_line_txt(filter_lines,raw_corpus_filter_path)

    return raw_corpus_filter_path


def drop_duplicates_lines(raw_corpus_filter_path):
    raw_corpus_filter = read_txt(raw_corpus_filter_path)
    raw_corpus_filter_df = pd.DataFrame({'text':raw_corpus_filter})

    # print(raw_corpus_filter_df.shape)
    raw_corpus_filter_df.drop_duplicates(subset=['text'],keep='first',inplace=True)
    # print(raw_corpus_filter_df.shape)
    filter_lines = raw_corpus_filter_df.text.values
    write_line_txt(filter_lines,raw_corpus_filter_path)
