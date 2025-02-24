import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

import utility.data_preprocessing as dpp
import utility.functions as fc
import torch
from transformers import AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

from datasets import Dataset

main_dir = "."
data_filename = "expected_primary_labels.xlsx"
sheet_names = ['Sheet1']
data_path = os.path.join(main_dir, "Data", data_filename)


def get_embedding(text, tokenizer, model):
    if isinstance(text, pd.Series) or isinstance(text, np.ndarray):  
        text = text.tolist()

    if not isinstance(text, list):  
        text = [text] 

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    embeddings = torch.mean(torch.stack(hidden_states[-1:]), dim=0)

    return embeddings


def extract_topic_cosine(df, embeddings, model_selected, exp_name=None):

    finetune_dir = fc.finetune_dir_path(exp_name, model_selected)
    tokenizer, _, _ = fc.setting_tokenizer(finetune_dir, model_selected)
    model = AutoModelForSequenceClassification.from_pretrained(finetune_dir)

    # Embedding expected labels' description: expected_labels['text']
    expected_labels = dpp.make_dataset(data_path, sheet_names)
    expected_labels['text'] = expected_labels['text'].apply(lambda x: str(x).replace("\xa0", "").lower())
    label_embeddings = get_embedding(expected_labels['text'].tolist(), tokenizer, model).cpu().numpy()
    label_embeddings = label_embeddings.mean(axis=1)

    cluster_groups = df.groupby("cluster")["text"].apply(list).to_dict()

    predicted_topics = {}

    for cluster_id, texts in cluster_groups.items():
        if len(texts) < 2:
            continue
        
        cluster_embeddings = get_embedding(texts, tokenizer, model).cpu().numpy()
        cluster_emb_avg = cluster_embeddings.mean(axis=0)

        # print(f"cluster_emb_avg.shape: {cluster_emb_avg.shape}")
        # print(f"label_embeddings.shape: {label_embeddings.shape}")
        similarities = cosine_similarity(cluster_emb_avg, label_embeddings)[0]
        
        best_label_idx = similarities.argmax()
        best_label = expected_labels['labels'].iloc[best_label_idx]
        
        predicted_topics[cluster_id] = best_label

    df['predicted_topics'] = df['cluster'].map(predicted_topics).fillna('Unknown')

    print(f"\t{predicted_topics}")
    return df


def extract_topic(df, embeddings, model_selected, exp_name=None):

    finetune_dir = fc.finetune_dir_path(exp_name, model_selected)
    tokenizer, _, _ = fc.setting_tokenizer(finetune_dir, model_selected)
    model = AutoModelForSequenceClassification.from_pretrained(finetune_dir)

    # Embedding expected labels' description: expected_labels['text']
    expected_labels = dpp.make_dataset(data_path, sheet_names)
    label_embeddings = get_embedding(expected_labels['text'].tolist(), tokenizer, model).cpu().numpy()
    label_embeddings = label_embeddings.mean(axis=1)

    cluster_groups = df.groupby("cluster")["text"].apply(list).to_dict()
    predicted_topics = {}
    cluster_topics = {}

    for cluster_id, texts in cluster_groups.items():
        if len(texts) < 2:
            print(f"  Skipping Cluster {cluster_id} (too few documents)...")
            continue
        print(f"Processing Cluster {cluster_id} with {len(texts)} documents...")

        combined_text = " ".join(texts)

        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Error: embeddings is empty or None!")

        if isinstance(embeddings, np.ndarray):
            pass
        elif hasattr(embeddings, "toarray"):  
            embeddings = embeddings.toarray() 
        else:
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

        # cluster_indices = df[df["cluster"] == cluster_id].index
        # if len(cluster_indices) == 0:
        #     raise ValueError("Error: cluster_indices is empty!")

        # cluster_embeddings = embeddings[cluster_indices]
        
        n_neighbors = min(len(texts) - 1, 5)
        umap_model = UMAP(n_components=5, n_neighbors=n_neighbors, metric='cosine', random_state=42)

        hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=2, gen_min_span_tree=True)
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)

        cluster_embeddings = get_embedding(combined_text, tokenizer, model).cpu().numpy()
        cluster_embeddings = cluster_embeddings.squeeze(0)

        
        # try:
        #     topics, probs = topic_model.fit_transform(texts, cluster_embeddings)
        #     if topics is None or len(topics) == 0:
        #         print(f"Skipping Cluster {cluster_id} (No topics found)...")
        #         continue

        #     df.loc[df["cluster"] == cluster_id, "topic"] = topics
        #     df.loc[df["cluster"] == cluster_id, "topic_confidence"] = probs
        #     cluster_topics[cluster_id] = topic_model.get_topic_info()
        # except Exception as e:
        #     print(f"Error in Cluster {cluster_id}: {e}")

        # df["topic"] = topics
        # df["topic_confidence"] = probs


        cluster_emb_avg = np.mean(cluster_embeddings, axis=0)

        similarities = cosine_similarity(cluster_emb_avg, label_embeddings)[0]

        best_label_idx = similarities.argmax()
        best_label = expected_labels['labels'].iloc[best_label_idx]

        predicted_topics[cluster_id] = best_label

    print("----------------------------------")
    print(predicted_topics)
    df['predicted_topics'] = df['topic'].map(predicted_topics).fillna('Unknown')
    print(df.head())
    return df



def extract_topic_old(df, embeddings, count_label, model_selected, exp_name=None):

    finetune_dir = fc.finetune_dir_path(exp_name)
    tokenizer, _, _ = fc.setting_tokenizer(finetune_dir, model_selected)
    model = AutoModelForSequenceClassification.from_pretrained(finetune_dir)

    # Embedding expected labels' description: expected_labels['text']
    expected_labels = dpp.make_dataset(data_path, sheet_names)

    label_texts = expected_labels['text'].tolist()
    label_embeddings = get_embedding(label_texts, tokenizer, model).cpu().numpy()
    label_embeddings = label_embeddings.mean(axis=1)

    # Grouping clustering outputs
    cluster_groups = df.groupby("cluster")["text"].apply(list).to_dict()
    cluster_topics = {}
    predicted_topics = {}
    
    for cluster_id, texts in cluster_groups.items():
        if len(texts) < 2:
            print(f"  Skipping Cluster {cluster_id} (too few documents)...")
            continue
        print(f"Processing Cluster {cluster_id} with {len(texts)} documents...")

        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("Error: embeddings is empty or None!")

        if isinstance(embeddings, np.ndarray):
            pass
        elif hasattr(embeddings, "toarray"):  
            embeddings = embeddings.toarray() 
        else:
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

        cluster_indices = df[df["cluster"] == cluster_id].index
        if len(cluster_indices) == 0:
            raise ValueError("Error: cluster_indices is empty!")

        cluster_embeddings = embeddings[cluster_indices]

        n_neighbors = min(len(texts) - 1, 5)
        umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=2, gen_min_span_tree=True)
        topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)

        try:
            topics, probs = topic_model.fit_transform(texts, cluster_embeddings)
            if topics is None or len(topics) == 0:
                print(f"Skipping Cluster {cluster_id} (No topics found)...")
                continue

            df.loc[df["cluster"] == cluster_id, "topic"] = topics
            df.loc[df["cluster"] == cluster_id, "topic_confidence"] = probs
            cluster_topics[cluster_id] = topic_model.get_topic_info()
        except Exception as e:
            print(f"Error in Cluster {cluster_id}: {e}")

    for cluster_id, topic_info in cluster_topics.items():
        valid_topics = topic_info[topic_info['Topic'] != -1]
        if not valid_topics.empty:
            top_topics = valid_topics.nlargest(2, 'Count')
        else:
            top_topics = topic_info.nlargest(2, 'Count')

        representations = []

        if len(top_topics) == 1:
            rep = top_topics['Representation'].iloc[0]
            if isinstance(rep, list) and len(rep) > 1:
                representations.extend([rep[0], rep[1]])
            elif isinstance(rep, list) and len(rep) == 1:
                representations.append(rep[0])
            else:
                representations.append(rep)
        else:
            representations = top_topics['Representation'].apply(lambda x: x[0] if isinstance(x, list) else x).tolist()

        topic_embeddings = get_embedding(representations, tokenizer, model).cpu().numpy()
        topic_emb_avg = topic_embeddings.mean(axis=0)
        topic_emb_avg = topic_emb_avg.mean(axis=0).reshape(1, -1)

        similarities = cosine_similarity(topic_emb_avg, label_embeddings)[0]
        best_label_idx = similarities.argmax()
        best_label = list(expected_labels['labels'])[best_label_idx]

        predicted_topics[cluster_id] = best_label

    print("----------------------------------")
    print(predicted_topics)

    df['predicted_topics'] = df['cluster'].apply(lambda cluster: predicted_topics.get(cluster, ''))
    return df
