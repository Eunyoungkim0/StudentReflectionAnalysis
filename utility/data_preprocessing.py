import numpy as np
import pandas as pd
import torch

#!pip install nltk
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

pre_stop_words = {'honestly', 'mainly', 'particularly', 'notable', 'probably', 'somewhat', 'really', 'significant', 'pretty'}

stop_words = set(stopwords.words('english'))
words_to_exclude = {'no', 'nor', 'not'}
stop_words = stop_words - words_to_exclude
words_to_include = {'would'}
stop_words.update(words_to_include)

import re

def remove_stopwords(text, stop_words):
    text = text.lower()
    filtered_words = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_words)


def make_dataset(data, sheet_name_list):
    df_ref = pd.read_excel(data, sheet_name=sheet_name_list[0])
    for sheet in sheet_name_list[1:]:
        temp_df = pd.read_excel(data, sheet_name=sheet)
        df_ref = pd.concat([df_ref, temp_df], ignore_index=True)

    return df_ref


from sklearn.preprocessing import LabelEncoder

def convert_to_categorical_data(df, column_answer):
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df[column_answer])
    unique_labels = list(label_encoder.classes_)
    count_label = len(label_encoder.classes_)

    return unique_labels, count_label


from sklearn.utils.class_weight import compute_class_weight

# For class weight adjustment due to class imbalance problem
def get_class_weights(df, column_name):
    labels = df[column_name].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    return class_weights_tensor


def data_preprocessing(df, column_input, column_answer, label_exist=False, test_data_labeled=True, train=False, train_labels=None):
    label_list = ['Python and Coding', 'MySQL', 'Github', 'API', 'Group Work', 'Time Management and Motivation', 'Other', 'None']
    df = df.rename(columns={column_input: 'original_text', column_answer: 'original_labels'})
    if label_exist == True:
        df = df[['original_text', 'original_labels', 'cluster', 'predicted_label']]
    elif test_data_labeled == False:
        df = df[['original_text', 'original_labels', 'predicted_label_BERT']]
    else:
        df = df[['original_text', 'original_labels']]

    df['text'] = df['original_text']
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].str.replace('||', '\n', regex=False)
    # df['text'] = df['text'].apply(remove_stopwords, args=(pre_stop_words,))
    df['text'] = df['text'].apply(lambda x: str(x).replace("\xa0", "").lower())
    df['text'] = df['text'].apply(remove_stopwords, args=(stop_words,))
    df['text'] = df['text'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))
    df['text'] = df['text'].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", " ", x))
    # df['original_labels'] = df['original_labels'].apply(lambda x: x if x in label_list else 'Other')
    # df['original_labels'] = df['original_labels'].apply(lambda x: 'No Issues' if x == 'None' else x)
    # df = df.dropna(subset=['original_labels'])

    if train == True:
        # If the data is less than 10 it converts the label to 'Other'
        df['original_labels'] = df['original_labels'].fillna('None')
        # label_counts = df['original_labels'].value_counts()
        # df['original_labels'] = df['original_labels'].apply(lambda x: 'Other' if label_counts[x] < 10 else x)

    if train_labels is not None:
        df.loc[~df['original_labels'].isin(train_labels), 'original_labels'] = 'Other'

    print(f"Number of rows in df (Final): {df.shape[0]}")

    unique_labels, count_label = convert_to_categorical_data(df, 'original_labels')
    class_weights_tensor = get_class_weights(df, 'labels')
    
    if train == True:
        return df, count_label, class_weights_tensor, unique_labels
    else:
        return df, count_label, class_weights_tensor