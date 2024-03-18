# !pip install sklearn_crfsuite
# !pip install seqeval
import pandas as pd
from sklearn.model_selection import KFold
import random
import csv
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, accuracy_score, f1_score
import random
random.seed(42)
def convert_to_bio_format(seq):
    # Apply postprocessing to each element in the array
    bio_sequence = (len(seq))*["O"]
    inside_entity = False
    # Find the start and end indices of the gold entity
    for i, token in enumerate(seq):
        if token.startswith("@@") and token.endswith("##"):
            bio_sequence[i] = "B"
            inside_entity = False
        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1

            # Find the index of the next token starting with @@
            for j, t in enumerate(seq[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break

            # Find the index of the next token ending with ##
            for j, t in enumerate(seq[i + 1:]):
                if t.endswith("##"):
                    next_hash_index = i + 1 + j
                    break

            if next_at_index != -1 and next_hash_index != -1 and next_hash_index < next_at_index:
                inside_entity = True
            if next_at_index == -1 and next_hash_index != -1:
                inside_entity = True

        elif inside_entity:
            bio_sequence[i] = "I"
            if not token.endswith("##"):
                inside_entity = True
            else:
                inside_entity = False
    return bio_sequence

def read(file_path, column_name):
    data = []
    with open(file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row[column_name])
    return data

train_files = [
    "./TOKENIZED_BIO_Economics_annotated_gold.csv",
    "./TOKENIZED_BIO_ukpolitics_annotated_gold.csv",
    "./TOKENIZED_BIO_politics_annotated_gold.csv"
]
test_file = "./TOKENIZED_BIO_Libertarian_annotated_gold.csv"
validation_file = "./TOKENIZED_BIO_worldpolitics_annotated_gold.csv"


train_X = []
train_Y = []
for file in train_files:
    print(file)
    X = read(file, "body_cleaned")
    Y = read(file, "gold_annotation")
    train_X.extend(X)
    train_Y.extend(Y)
#train_Y_bio = convert_to_bio_format(train_Y)
# Concatenate data for validation
validation_X = read(validation_file, "body_cleaned")
validation_Y = read(validation_file, "gold_annotation")
#validation_Y_bio = convert_to_bio_format(validation_Y)
# Data for testing
test_X = read(test_file, "body_cleaned")
test_Y = read(test_file, "gold_annotation")
#test_Y_bio = convert_to_bio_format(test_Y)


def convert_to_bio_format(seq):
    # Apply postprocessing to each element in the array
    bio_sequence = (len(seq))*["O"]
    inside_entity = False
    # Find the start and end indices of the gold entity
    for i, token in enumerate(seq):
        if token.startswith("@@") and token.endswith("##"):
            bio_sequence[i] = "B"
            inside_entity = False
        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1

            # Find the index of the next token starting with @@
            for j, t in enumerate(seq[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break

            # Find the index of the next token ending with ##
            for j, t in enumerate(seq[i + 1:]):
                if t.endswith("##"):
                    next_hash_index = i + 1 + j
                    break

            if next_at_index != -1 and next_hash_index != -1 and next_hash_index < next_at_index:
                inside_entity = True
            if next_at_index == -1 and next_hash_index != -1:
                inside_entity = True

        elif inside_entity:
            bio_sequence[i] = "I"
            if not token.endswith("##"):
                inside_entity = True
            else:
                inside_entity = False
    return bio_sequence

# Feature extraction function
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True

    return features

# Feature extraction for entire sentence
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return sent.split()

# Convert training data to feature-label pairs
X_train = [sent2features(sent.split()) for sent in train_X]
y_train = [sent2labels(labels) for labels in train_Y]
y_train_bio = [convert_to_bio_format(labels) for labels in y_train]

X_validation = [sent2features(sent.split()) for sent in validation_X]
y_validation = [sent2labels(labels) for labels in validation_Y]
y_validation_bio = [convert_to_bio_format(labels) for labels in y_validation]

X_test = [sent2features(sent.split()) for sent in test_X]
y_test = [sent2labels(labels) for labels in test_Y]
y_test_bio = [convert_to_bio_format(labels) for labels in y_test]
# Initialize CRF model
crf = CRF(algorithm='pa', #l2sgd #ap #pa #arow #lbfgs
    #c1=0.1,
    #c2=0.1,
    c=0.5,
    max_iterations=100,
    all_possible_transitions=True)

# Train the model
crf.fit(X_train, y_train_bio)

# Make predictions on validation set
y_pred_validation = crf.predict(X_validation)

# Flatten the lists
validation_true_labels_flat = [label for sublist in y_validation_bio for label in sublist]
validation_predicted_labels_flat = [label for sublist in y_pred_validation for label in sublist]

print("Seqeval Classification Report: \n", classification_report(y_validation_bio, y_pred_validation, mode = "strict" ))
