import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

import utility.data_preprocessing as dpp
import utility.functions as fc


main_dir = "."
data_filename = "cluster_output_DistilBERT (with fine-tuning).xlsx"

data_path = os.path.join(main_dir, "clustering/result", data_filename)
# sheet_names = ['Spectral Clustering', 'Affinity Propagation Clustering', 'K-means Clustering']
sheet_names = ['K-means Clustering']


def extract_topic(df, embeddings):
    stop_words = {'excited', 'satisfied', 'frustrated', 'neutral', 'confused', 'anxious',
                #   'no', 
                  'not', 'nope', 'nothing', 'none', 'na', 
                   'much', 'far', 'yet', 'biggest', 'challenge', 'challenges', 'challenging', 'right', 'able', 'using'}
    df['text'] = df['text'].apply(dpp.remove_stopwords, args=(stop_words,))
    df['text'] = df['text'].apply(lambda x: x.replace('(No strong positive or negative emotions)', ''))
    cluster_groups = df.groupby("cluster")["text"].apply(list).to_dict()

    cluster_topics = {}
    predicted_topics = {}

    for cluster_id, texts in cluster_groups.items():
        if len(texts) < 2:
            print(f"  Skipping Cluster {cluster_id} (too few documents)...")
            continue
        print(f"Processing Cluster {cluster_id} with {len(texts)} documents...")

        cluster_indices = df[df["cluster"] == cluster_id].index
        cluster_embeddings = np.array(embeddings)[cluster_indices]

        # n_neighbors = min(5, len(texts) - 1)
        # topic_model = BERTopic(umap_model=UMAP(n_components=2, n_neighbors=n_neighbors, metric='cosine'))
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
            # print(df[['original_labels', 'predicted_label', 'topic', 'topic_confidence', 'text']])
        except Exception as e:
            print(f"Error in Cluster {cluster_id}: {e}")

    
    for cluster_id, topic_info in cluster_topics.items():
        # print(f"\nCluster {cluster_id} topics:")
        # print(topic_info[['Topic', 'Count', 'Name', 'Representation']])

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

        predicted_topics[cluster_id] = ", ".join(representations)

    print("----------------------------------")
    print(predicted_topics)

    df['predicted_topics'] = df['cluster'].apply(lambda cluster: predicted_topics.get(cluster, ''))
    return df

    # excel_df = processed_df[['original_text', 'original_labels', 'cluster', 'predicted_label', 'predicted_topics']]

    # with pd.ExcelWriter(data_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
    #     excel_df.to_excel(writer, index=True, sheet_name='K-means Clustering')


if __name__ == "__main__":
    str_reflection = ""
    str_embedding_method = ""
    str_clustering_method = ""

    while True:
        try:
            embedding_method = int(input(
                "\nChoose embedding methods: \n"
                "\t1: TF-IDF, \n"
                "\t2: Sentence Transformers, \n"
                "\t3: DistilBERT without finetuning, \n"
                "\t4: DistilBERT with finetuning, \n"
                "\t0: Exit \n"
            ))

            if embedding_method == 0:
                print("Exiting the program.")
                break
            elif embedding_method in (1, 2, 3, 4):
                print(f"You selected method {embedding_method}.")
                break
            else:
                print("Invalid input. Please choose a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 1. Load data
    processed_df, count_label, class_weights_tensor = fc.load_data(data_path, sheet_names, "original_text", "original_labels", True)
    processed_df['original_labels'] = processed_df['original_labels'].fillna('None')
    processed_df['predicted_label'] = processed_df['predicted_label'].fillna('None')

    # 2. Generate Embeddings
    embeddings, str_embedding_method = fc.generate_embeddings(embedding_method, processed_df, count_label)

    # 3. Extract Topic
    extract_topic(processed_df, embeddings)