import pandas as pd
import numpy as np
# from algorithms import *
# from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import re
import math
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle


def AddSpamData(current_csv_file='vietnamese_data.csv', spam_text_file='spams.txt', save_file=True):
    '''
        This function appends data in a text file containing spam message in each line to a csv file which has two columns named 'text' and 'spam', and returns the pandas dataframe corresponding to the data after concatenation
            + The names of the files are specified in the current_csv_file and spam_text_file parameters
            + If save_file is set to True, the function will save the current_csv_file with the new dataframe
    '''
    df_current = pd.read_csv(current_csv_file)
    spams = []
    with open(spam_text_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            spams.append(line[:-1])
    labels = np.zeros((len(spams,))).astype(int)
    df_spam = pd.DataFrame({'text':spams, 'spam':labels})

    df_sum = pd.concat([df_current, df_spam], ignore_index=True)
    df_sum.reset_index()
    df_sum['text'] = df_sum['text'].astype(str)

    if save_file:
        df_sum.to_csv(current_csv_file, index=False)
    return df_sum

def count_weird_char(text:str):
    '''
    This function counts the number of non-vietnamese characters of a text string, creating a new feature for the dataset
    This feature may help detect spam cases like these:
        + Nhung c0 gäi xjnh dep, phUc vu tinh dUc tren toän Vjjet Näm, giä cä uu däi Ljen he Zäl.0: .568653079  zap
        + t.o ta.i kh.oanta.ng ba.n 50k-1555k  https://by.tn/LYz8
    '''
    vietnamese_pattern = r'[0-9a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý\s|_]' # If we don't need special ascii characters, we can add \W
    vietnamese_chars = re.compile(vietnamese_pattern)
    # Count the non-Vietnamese characters using regex
    weird_chars = vietnamese_chars.sub('', text)
    # print(weird_chars)
    return len(weird_chars)

def maximum_token_length(text:str):
    '''
    This function returns the maximum token length of a text string, creating a new feature for the dataset
    This feature may help detect spam cases like these:
        + U  V S  . CongViecDeDang Lien(He)Zalo . Luong500-3000k/Ngay
    '''
    return len(max(text.split(), key=lambda item:len(item)))

def count_links(text:str):
    link_pattern = r'[a-zA-Z0-9]+\.[a-z]+\/([a-zA-Z0-9]+)?|[a-zA-Z0-9]+\.com|https?:\/\/[a-zA-Z0-9\.\/]+|www.[a-zA-Z0-9\.\/]+'
    pattern = re.compile(link_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count

def count_phone_numbers(text:str):
    phone_pattern = r'\+?[0-9]{11,12}|0[0-9]{3}[\. -]?[0-9]{3}[\. -]?[0-9]{3}|0[0-9]{3}[\. -]?[0-9]{2}[\. -]?[0-9]{2}[\. -]?[0-9]{2}|0[0-9]{2}[\. -]?[0-9]{3}[\. -]?[0-9]{4}|(18|19)00[0-9]{4}'
    pattern = re.compile(phone_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count

def count_weird_capitalization(text:str):
    cap_pattern = r'[a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểếễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹý]*[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý]*'
    pattern = re.compile(cap_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count

def count_money(text:str):
    money_pattern = r'([0-9]+[\.,])*[0-9]+([Kkdđ]| ngàn| nghìn|.ooo|tr| triệu| vnd| ty| tỷ| đồng| tỉ| ti)'
    pattern = re.compile(money_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count

def capitalization_proportion(text: str):
    cap_pattern = r'[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][0-9a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý]*'
    pattern = re.compile(cap_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count / len(text.split())

def import_stop_words(filename = 'vietnamese-stopwords.txt'):
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        stop_words = []
        for line in lines:
            token = ViTokenizer.tokenize(line)
            token = token.split()
            stop_words.extend(token)
    punctuations = ['!','"',".",",",":",'?','\\','/','#','(',')','$','%','&','@','^','*','-','_','+','=','{','[',']','}','|',';','\'','<','>']
    stop_words = stop_words + punctuations
    return stop_words

def tokenize(text: str, stop_words):
    text = text.lower()
    tokenized_str = ViTokenizer.tokenize(text)
    lst = tokenized_str.split()
    tokens = []
    for word in lst:
        if word not in stop_words:
            if ("_" in word) or (word.isalpha() == True):
                tokens.append(word)
    sentence = ' '.join(tokens)
    return sentence

vect = TfidfVectorizer()
stop_words = import_stop_words()
over_sampling = SMOTEENN()
def preprocess_file(vectorizer=vect, filename='vietnamese_data.csv', stop_words = stop_words, over_sampling = over_sampling, test_size=0.25):
    dataset1_df = pd.read_csv(filename)
    dataset1_df = dataset1_df[dataset1_df.columns[:2]]
    df = dataset1_df.sample(frac = 1)
    x = df['text'].astype(str)
    X = []
    additional_features = []
    for sentence in x:
        tokenized_sent = tokenize(sentence, stop_words)
        X.append(tokenized_sent)
        additional_features.append([count_weird_char(sentence), maximum_token_length(sentence), count_links(sentence), 
                                    count_phone_numbers(sentence), count_weird_capitalization(sentence), count_money(sentence),
                                    capitalization_proportion(sentence)])
    X_array = np.array(additional_features)
    split_index = math.floor(len(X)*(1-test_size))
    X_train_v2 = X_array[:split_index]
    X_test_v2 = X_array[split_index:]
    y = df['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train_v1 = vectorizer.fit_transform(X_train)
    X_test_v1 = vectorizer.transform(X_test)
    X_train_v = np.concatenate((X_train_v1.toarray(), X_train_v2), axis=1)
    X_test_v = np.concatenate((X_test_v1.toarray(), X_test_v2), axis=1)
    X_train_smenn, y_train_smenn = over_sampling.fit_resample(X_train_v, y_train)
    return X_train_smenn, X_test_v, y_train_smenn, y_test

def preprocess_train(vectorizer=vect, filename='vietnamese_data.csv', stop_words = stop_words, over_sampling = over_sampling):
    df = pd.read_csv(filename)
    df = df[df.columns[:2]]
    x = df['text'].astype(str)
    y_train = df['spam']
    X1 = []
    additional_features = []
    for sentence in x:
        tokenized_sent = tokenize(sentence, stop_words)
        X1.append(tokenized_sent)
        additional_features.append([count_weird_char(sentence), maximum_token_length(sentence), count_links(sentence), 
                                    count_phone_numbers(sentence), count_weird_capitalization(sentence), count_money(sentence),
                                    capitalization_proportion(sentence)])
    Fit = vectorizer.fit(X1)
    filename = 'finalized_vectorizer.sav'
    pickle.dump(Fit, open(filename, 'wb'))
    X_train_v1 = Fit.transform(X1)
    X_train_v2 = np.array(additional_features)
    X_train_v = np.concatenate((X_train_v1.toarray(), X_train_v2), axis=1)
    X_train_smenn, y_train_smenn = over_sampling.fit_resample(X_train_v, y_train)
    return filename, X_train_smenn, y_train_smenn

def preprocess_list(test_list:list, Fit):
    X_test1 = []
    additional_features = []
    for test_string in test_list:
        X_test1.append(tokenize(test_string, stop_words))
        additional_features.append([count_weird_char(test_string), maximum_token_length(test_string), count_links(test_string), 
                                        count_phone_numbers(test_string), count_weird_capitalization(test_string), count_money(test_string),
                                        capitalization_proportion(test_string)])
    X_test_v1 = Fit.transform(X_test1)
    X_test_v2 = np.array(additional_features)
    X_test_v = np.concatenate((X_test_v1.toarray(), X_test_v2), axis=1)
    return X_test_v

def preprocess_test_output(test_output):
    return np.array(test_output)

if __name__ == '__main__':
    Fit, X_train_v, y_train = preprocess_train()
    l = []
    while True:
        s = input()
        if s == '*':
            break
        else:
            l.append(s)
    X_test_v = preprocess_list(l, Fit)
    print(X_test_v)

