import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# with open("./bbc-text.csv", "r") as csvfile:
#     reader = csv.reader(csvfile, delimiter='\n')
#     for line in reader:
#         print(line)


# GRADED FUNCTION: remove_stopwords
def remove_stopwords(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (string): sentence to remove the stopwords from

    Returns:
        sentence (string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    # Sentence converted to lowercase-only
    sentence = sentence.lower()

    ### START CODE HERE
    sentence_removed_stopword = []
    for word in sentence.split():
        if word not in stopwords:
            sentence_removed_stopword.append(word)
    sentence = " ".join([word for word in sentence_removed_stopword])
    ### END CODE HERE
    return sentence


# GRADED FUNCTION: parse_data_from_file
def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file

    Args:
        filename (string): path to the CSV file

    Returns:
        sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """
    sentences = []
    labels = []
    with open(filename, 'r') as csvfile:
        ### START CODE HERE
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            labels.append(row[2])
            sentences.append(remove_stopwords(row[1]))
        ### END CODE HERE
    return sentences, labels

sentences, labels = parse_data_from_file("./bbc-text.csv")

print("ORIGINAL DATASET:\n")
print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}\n\n")


def fit_tokenizer(sentences):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer


tokenizer = fit_tokenizer(sentences)
word_index = tokenizer.word_index

print(f"Vocabulary contains {len(word_index)} words\n")
print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")


def get_padded_sequences(tokenizer, sentences):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding="post")
    return padded_sequences

padded_sequences = get_padded_sequences(tokenizer, sentences)
print(f"First padded sequence looks like this: \n\n{padded_sequences[0]}\n")
print(f"Numpy array of all sequences has shape: {padded_sequences.shape}\n")
print(f"This means there are {padded_sequences.shape[0]} sequences in total and each one has a size of {padded_sequences.shape[1]}")

def tokenize_labels(labels):
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    label_sequences = label_tokenizer.texts_to_sequences(labels)
    return label_sequences, label_word_index


label_sequences, label_word_index = tokenize_labels(labels)
print(f"Vocabulary of labels looks like this {label_word_index}\n")
print(f"First ten sequences {label_sequences[:10]}\n")