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
    labels = np.ones((len(spams, ))).astype(int)
    df_spam = pd.DataFrame({'text': spams, 'spam': labels})

    df_sum = pd.concat([df_current, df_spam], ignore_index=True)
    df_sum.reset_index()
    df_sum['text'] = df_sum['text'].astype(str)

    if save_file:
        df_sum.to_csv(current_csv_file, index=False)
    return df_sum


def count_weird_char(text: str):
    '''
    This function counts the number of non-vietnamese characters of a text string, creating a new feature for the dataset
    This feature may help detect spam cases like these:
        + Nhung c0 gäi xjnh dep, phUc vu tinh dUc tren toän Vjjet Näm, giä cä uu däi Ljen he Zäl.0: .568653079  zap
        + t.o ta.i kh.oanta.ng ba.n 50k-1555k  https://by.tn/LYz8
    '''
    vietnamese_pattern = r'[0-9a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý\s|_]'  # If we don't need special ascii characters, we can add \W
    vietnamese_chars = re.compile(vietnamese_pattern)
    # Count the non-Vietnamese characters using regex
    weird_chars = vietnamese_chars.sub('', text)
    # print(weird_chars)
    return len(weird_chars)


def maximum_token_length(text: str):
    '''
    This function returns the maximum token length of a text string, creating a new feature for the dataset
    This feature may help detect spam cases like these:
        + U  V S  . CongViecDeDang Lien(He)Zalo . Luong500-3000k/Ngay
    '''
    return len(max(text.split(), key=lambda item: len(item)))


def count_links(text: str):
    link_pattern = r'[a-zA-Z0-9]+\.[a-z]+\/([a-zA-Z0-9]+)?|[a-zA-Z0-9]+\.com|https?:\/\/[a-zA-Z0-9\.\/]+|www.[a-zA-Z0-9\.\/]+'
    pattern = re.compile(link_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count


def count_phone_numbers(text: str):
    phone_pattern = r'\+?[0-9]{11,12}|0[0-9]{3}[\. -]?[0-9]{3}[\. -]?[0-9]{3}|0[0-9]{3}[\. -]?[0-9]{2}[\. -]?[0-9]{2}[\. -]?[0-9]{2}|0[0-9]{2}[\. -]?[0-9]{3}[\. -]?[0-9]{4}|(18|19)00[0-9]{4}'
    pattern = re.compile(phone_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count


def count_weird_capitalization(text: str):
    cap_pattern = r'[a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽềềểếễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹý]*[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý]*'
    pattern = re.compile(cap_pattern)
    matches = pattern.finditer(text)
    count = 0
    for match in matches:
        # print(match.group(0))
        count += 1
    return count


def count_money(text: str):
    money_pattern = r'([0-9]+[\.,])*[0-9]+([Kkdđ]| ngàn| nghìn|.ooo|tr| triệu| vnd| ty| tỷ| tỉ| đồng)'
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


def import_stop_words(filename='vietnamese-stopwords.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        stop_words = []
        for line in lines:
            token = ViTokenizer.tokenize(line)
            token = token.split()
            stop_words.extend(token)
    punctuations = ['!', '"', ".", ",", ":", '?', '\\', '/', '#', '(', ')', '$', '%', '&', '@', '^', '*', '-', '_', '+',
                    '=', '{', '[', ']', '}', '|', ';', '\'', '<', '>']
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


def preprocess_file(vectorizer=vect, filename='vietnamese_data.csv', stop_words=stop_words, over_sampling=over_sampling,
                    test_size=0.25):
    dataset1_df = pd.read_csv(filename)
    dataset1_df = dataset1_df[dataset1_df.columns[:2]]
    df = dataset1_df.sample(frac=1)
    x = df['text'].astype(str)
    X = []
    additional_features = []
    for sentence in x:
        tokenized_sent = tokenize(sentence, stop_words)
        X.append(tokenized_sent)
        additional_features.append([count_weird_char(sentence), maximum_token_length(sentence), count_links(sentence),
                                    count_phone_numbers(sentence), count_weird_capitalization(sentence),
                                    count_money(sentence),
                                    capitalization_proportion(sentence)])
    X_array = np.array(additional_features)
    split_index = math.floor(len(X) * (1 - test_size))
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


def preprocess_train(vectorizer=vect, filename='vietnamese_data.csv', stop_words=stop_words,
                     over_sampling=over_sampling):
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
                                    count_phone_numbers(sentence), count_weird_capitalization(sentence),
                                    count_money(sentence),
                                    capitalization_proportion(sentence)])
    Fit = vectorizer.fit(X1)
    X_train_v1 = Fit.transform(X1)
    X_train_v2 = np.array(additional_features)
    X_train_v = np.concatenate((X_train_v1.toarray(), X_train_v2), axis=1)
    X_train_smenn, y_train_smenn = over_sampling.fit_resample(X_train_v, y_train)
    return Fit, X_train_smenn, y_train_smenn


def preprocess_test_str(test_string, Fit):
    X_test1 = [tokenize(test_string, stop_words)]
    additional_features = [[count_weird_char(test_string), maximum_token_length(test_string), count_links(test_string),
                            count_phone_numbers(test_string), count_weird_capitalization(test_string),
                            count_money(test_string),
                            capitalization_proportion(test_string)]]
    X_test_v1 = Fit.transform(X_test1)
    X_test_v2 = np.array(additional_features)
    X_test_v = np.concatenate((X_test_v1.toarray(), X_test_v2), axis=1)
    return X_test_v


def preprocess_string(test_string, test_output, Fit):
    X_test_v = preprocess_test_str(test_string, Fit)
    y_test = np.array([test_output])
    return X_test_v, y_test


import tensorflow as tf
import matplotlib.pyplot as plt


# Define the necessary Activation Function
# 1. ReLU(Rectified Linear Unit)
def relu(Z):
    relu_Z = tf.math.greater_equal(Z, tf.constant([0.]))
    # Convert relu_Z to type float64 since tf.multiply doesn't accept type Boolean
    relu_Z = tf.cast(relu_Z, dtype=tf.float32)
    return tf.math.multiply(relu_Z, Z)


# 2. Sigmoid
def sigmoid(Z):
    return tf.cast(tf.math.sigmoid(Z), dtype=tf.float32)


# 3. Tanh
def tanh(Z):
    return tf.cast(tf.math.tanh(Z), dtype=tf.float32)


# 4. Softmax
def softmax(Z):
    return tf.cast(tf.nn.softmax(Z), dtype=tf.float32)


# 5. Compute derivatives of a general Activation Function
def derivative(Z, activation):
    with tf.GradientTape() as g:
        g.watch(Z)
        Y = activation(Z)
    dy_dz = g.gradient(Y, Z)
    return dy_dz


class Layer:
    def __init__(self, W, b, activation):
        self.W = W
        self.b = b
        self.activation = activation


class ANN:
    def __init__(self, X, T):
        # Get the train data
        self.X = X
        self.T = T  # Expect T haven't been one-hot-encoded
        self.Y = None  # Y => predict output
        self.N = len(X)  # The number of samples
        self.D = X.shape[1]  # The number of features of X
        self.K = len(set(T))  # The number of classes to predict
        # Define the number of layers, weight lists and layer nodes
        self.num_layers = 0
        self.layer_list = []
        self.hidden_nodes = [tf.convert_to_tensor(self.X,
                                                  dtype=tf.float32)]  # Keep the hidden nodes list for future use (backpropagation with recursion delta)

    # Define the one-hot-encoding and one-hot-decoding function
    def one_hot_encoding(self, Z):  # Expect Z as numpy 1D-array
        K = len(set(Z))
        N = len(Z)
        Z2 = np.zeros((N, K))
        for n in range(N):
            t = int(Z[n])
            Z2[n, t] = 1
        return Z2

    def one_hot_decoding(self, Z):  # Expect Z as a one-hot-encoding numpy array
        return np.argmax(Z, axis=1)

    # Define the function to allow add layers for the Network
    def add_layer(self, hidden_size=5, activation=relu):
        if self.num_layers == 0:
            # Define random weights and biases for input layer->first hidden layer
            W = tf.random.normal(shape=[self.D, hidden_size], dtype=tf.float32)
            W = W / tf.reduce_max(tf.math.abs(W))
            b = tf.random.normal(shape=[1, hidden_size], dtype=tf.float32)
            b = b / tf.reduce_max(tf.math.abs(b))
            # Create new layer
            new_layer = Layer(W, b, activation)
            self.hidden_nodes.append(
                activation(tf.add(tf.matmul(tf.convert_to_tensor(self.X, dtype=tf.float32), W), b)))
            self.layer_list.append(new_layer)
            self.num_layers += 1
        else:
            previous_hidden_size = self.hidden_nodes[-1].shape[1]
            # Define random weights and biases for last current hidden layer-> new hidden layer or output layer
            W = tf.random.normal(shape=[previous_hidden_size, hidden_size], dtype=tf.float32)
            W = W / tf.reduce_max(tf.math.abs(W))
            b = tf.random.normal(shape=[1, hidden_size], dtype=tf.float32)
            b = b / tf.reduce_max(tf.math.abs(b))
            # Create new layer
            new_layer = Layer(W, b, activation)
            self.hidden_nodes.append(activation(tf.add(tf.matmul(self.hidden_nodes[-1], W), b)))
            self.layer_list.append(new_layer)
            self.num_layers += 1

    # Define the forward function
    def forward(self, X):
        Z = tf.convert_to_tensor(X, dtype=tf.float32)
        for i in range(self.num_layers):
            layer = self.layer_list[i]
            Z = layer.activation(tf.add(tf.matmul(Z, layer.W), layer.b))
            self.hidden_nodes[i + 1] = Z
        return Z

    # Define the cost function:
    def cost(self, X, T, C=0.01):
        Y = self.forward(X)  # Get the predicted output
        regularization_term = 0
        for layer in self.layer_list:
            regularization_term += tf.reduce_sum(layer.W).numpy()
        return tf.reduce_sum(1 / self.N * tf.math.multiply(T, tf.math.log(
            Y))).numpy() - C * 1 / self.N * regularization_term  # Cost = T.log(Y)

    # Define the predict function
    def classification_rate(self, Y,
                            T):  # Expect Y is a one-hot-encoding tensor, T is the correct output as numpy 1D-array
        Y = Y.numpy()  # convert Y to numpy array
        Y = self.one_hot_decoding(Y)  # Decode Y
        n_correct = 0
        n_total = 0
        n_correct1 = 0
        for i in range(len(Y)):
            n_total += 1
            if Y[i] == T[i]:
                n_correct += 1
                if T[i] == 1:
                    n_correct1 += 1
        print("Spam Predict Correct:", n_correct1)
        print("Correct Prediction / Total mails:", (n_correct, n_total))
        return float(n_correct) / n_total

    # Define the backpropagtion function
    def backpropagation(self, learning_rate=1e-6, n_iters=400, C=0.01):
        tensor_T = tf.convert_to_tensor(self.one_hot_encoding(self.T),
                                        dtype=tf.float32)  # Convert the self.T to one-hot-encoding tensor

        losses = []
        for _ in range(n_iters):
            self.Y = self.forward(self.X)  # Feedforward to get the prediction output as Y
            # Define the gradient of Weights and gradient of biases
            grad_W = []
            grad_B = []

            # Implement Backpropagation => recursive delta
            delta = tf.subtract(tensor_T, self.Y)
            grad_w = tf.matmul(tf.transpose(self.hidden_nodes[-2]), delta) - C * self.layer_list[-1].W
            grad_b = tf.matmul(tf.ones(shape=(1, self.N), dtype=tf.float32), delta) - C * self.layer_list[-1].b

            grad_W.append(grad_w)
            grad_B.append(grad_b)

            for i in range(self.num_layers - 1, 0, -1):
                delta = tf.multiply(tf.matmul(delta, tf.transpose(self.layer_list[i].W)),
                                    derivative(self.hidden_nodes[i], self.layer_list[i - 1].activation))
                grad_w = tf.matmul(tf.transpose(self.hidden_nodes[i - 1]), delta) - C * self.layer_list[i - 1].W
                grad_b = tf.matmul(tf.ones(shape=(1, self.N), dtype=tf.float32), delta) - C * self.layer_list[i - 1].b

                grad_W.insert(0, grad_w)
                grad_B.insert(0, grad_b)

            # Gradient descent
            for i in range(self.num_layers):
                self.layer_list[i].W += learning_rate * grad_W[i]
                self.layer_list[i].b += learning_rate * grad_B[i]

            if _ % 100 == 0:
                loss = self.cost(self.X, tensor_T, C=C)
                losses.append(loss)
                Y = self.forward(self.X)
                print('Train Accuracy:', self.classification_rate(Y, self.T), "--loss:", loss)

        plt.plot(losses)
        plt.title("loss per iteration")
        plt.show()

        self.Y = self.forward(self.X)
        print('Train Accuracy:', self.classification_rate(self.Y, self.T))


Fit, X_train, y_train = preprocess_train()
model = ANN(X_train, y_train)
model.add_layer(hidden_size=5, activation=relu)
model.add_layer(hidden_size=2, activation=softmax)
model.backpropagation(n_iters=100, learning_rate=1e-5, C=0.01)

test_string = input()
test_output = int(input())
X_test, y_test = preprocess_string(test_string, test_output, Fit)

Z = model.forward(X_test)
print(Z)
model.classification_rate(Z, y_test)
