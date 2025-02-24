#----------------------------------------------------------------------------------------
# Setting up Some Variables
ref_num = 0
train_valid_split_yn = 'y'
test_data_labeled = 'n'
train_only = 'y'
exp_name = 'r1_100'

model_selected = 'DistilBERT'
# model_selected = 'BERT'
# model_selected = 'RoBERTa'

ln_rt = 7e-5
bt_sz = 24
epochs = 10
k_fold_number = 5

column_input = "text"
column_answer = "labels"
#----------------------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch

import utility.data_preprocessing as dpp
import utility.functions as fc

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

import shutil
from datasets import Dataset

main_dir = "."

# Data filename for each dataset
if(exp_name == "r1_100"):
    data_filename = "SL-R1-100P (2_21).xlsx"
    data_filename_valid = "SL-R1-100P (2_21).xlsx"
    train_sheet_names = ['sl-r1-100.csv']
    valid_sheet_names = ['sl-r1-100.csv']
elif(exp_name == "r1_80"):
    data_filename = "SL-R1-80P (2_21).xlsx"
    data_filename_valid = "SL-R1-80P (2_21).xlsx"
    train_sheet_names = ['sl-r1-80.csv']
    valid_sheet_names = ['sl-r1-80.csv']
elif(exp_name == "all_100"):
    data_filename = "SL-R-ALL-100-2_11.xlsx"
    data_filename_valid = "SL-R-ALL-100-2_11.xlsx"
    train_sheet_names = ['fixed_sl-all-100']
    valid_sheet_names = ['fixed_sl-all-100']
elif(exp_name == "all_80"):
    data_filename = "SL-R-ALL-80-2_11.xlsx"
    data_filename_valid = "SL-R-ALL-80-2_11.xlsx"
    train_sheet_names = ['fixed_sl-all-80']
    valid_sheet_names = ['fixed_sl-all-80']

data_filename_test = "Spring2025_reflection.xlsx"
test_sheet_names = ['SD-ESA5-1']

# Data path for Train/Validation/Test data
data_train = os.path.join(main_dir, "Data", data_filename)
data_valid = os.path.join(main_dir, "Data", data_filename_valid)
data_test = os.path.join(main_dir, "Data", data_filename_test)

# Directory for Results/Fine-tuned Model
result_dir = os.path.join(main_dir, "Results")
finetune_dir_name = "finetuned_" + model_selected + "_" + exp_name
finetune_dir = os.path.join(main_dir, "Models", finetune_dir_name)

# Delete the directories if they exist
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)

if os.path.exists(finetune_dir):
    shutil.rmtree(finetune_dir)


class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.class_weights = class_weights
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # labels = inputs.get("labels")
        labels = inputs.get("labels").to(self.model.device)
        # outputs = model(**inputs)
        outputs = model(**{k: v.to(self.model.device) for k, v in inputs.items()})
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
def k_fold_cross_validation(train, model, training_args, tokenizer, data_collator, class_weights_tensor, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        print(f"Training fold {fold + 1}/{k}")

        # train/validation split
        train_fold = train.select(train_idx)
        valid_fold = train.select(val_idx)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=valid_fold,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=fc.compute_metrics,
            class_weights=class_weights_tensor
        )
        
        trainer.train()

        eval_results = trainer.evaluate(valid_fold)
        fold_results.append(eval_results)
        
        # print(f"Results for fold {fold + 1}: {eval_results}")
        
    avg_results = {key: sum(result[key] for result in fold_results) / k for key in fold_results[0].keys()}
    # print(f"Average results across {k}-fold cross-validation: {avg_results}")
    return avg_results


start_time = datetime.now()
print(f"start time: {start_time}")

# 1. Load Data
if ref_num == 0:
    train_data = train_sheet_names
    valid_data = valid_sheet_names if train_valid_split_yn == 'y' else None
    test_data = test_sheet_names if test_data_labeled == 'n' else None
else:
    train_data = [train_sheet_names[ref_num-1]]
    valid_data = [valid_sheet_names[ref_num-1]] if train_valid_split_yn == 'y' else None
    test_data = [test_sheet_names[ref_num-1]] if test_data_labeled == 'n' else None

df_ref = dpp.make_dataset(data_train, train_data)
df_ref_valid = dpp.make_dataset(data_valid, valid_data) if valid_data else None
df_ref_test = dpp.make_dataset(data_test, test_data) if test_data else None


# 2. Data pre-processing and Tokenization
# 2-1. For Training/Validation 
if(train_valid_split_yn == 'y'):

    # Train data
    processed_df, count_label, class_weights_tensor, train_labels = dpp.data_preprocessing(df_ref, column_input, column_answer, train=True)
    dataset = Dataset.from_pandas(processed_df)
    # Test data
    processed_df_valid, _, _ = dpp.data_preprocessing(df_ref_valid, column_input, column_answer, train_labels)
    dataset_valid = Dataset.from_pandas(processed_df_valid)
    # Tokenization
    tokenized_dataset = dataset.map(fc.tokenizer_function, batched=True, fn_kwargs={'model_selected': model_selected, 'finetuned': False})
    tokenized_dataset_valid = dataset_valid.map(fc.tokenizer_function, batched=True, fn_kwargs={'model_selected': model_selected, 'finetuned': False})
    
    train = tokenized_dataset
    valid = tokenized_dataset_valid

else:
    # Data
    processed_df, count_label, class_weights_tensor, train_labels = dpp.data_preprocessing(df_ref, column_input, column_answer, train=True)
    dataset = Dataset.from_pandas(processed_df)
    # Tokenization
    tokenized_dataset = dataset.map(fc.tokenizer_function, batched=True, fn_kwargs={'model_selected': model_selected, 'finetuned': False})

    df_tk = pd.DataFrame(tokenized_dataset)
    label_counts = df_tk['labels'].value_counts()
    majority_df = df_tk[df_tk['labels'].map(lambda x: label_counts[x] > 1)]
    minority_df = df_tk[df_tk['labels'].map(lambda x: label_counts[x] == 1)]

    # stratify: Due to not enough data, I wanted to keep the data distribution the same.
    train_major, valid_major = train_test_split(majority_df, test_size=0.2, stratify=majority_df['labels'], random_state=42, shuffle=True)
    
    train_df = pd.concat([train_major, minority_df]).reset_index(drop=True)
    valid_df = valid_major.reset_index(drop=True)

    train = Dataset.from_pandas(train_df)
    valid = Dataset.from_pandas(valid_df)

# 2-2. For Testing
if(test_data_labeled == 'n'):
    # Data
    processed_df_test, _, _, train_labels = dpp.data_preprocessing(df_ref_test, column_input, column_answer, train=True)
    dataset_test = Dataset.from_pandas(processed_df_test)
    # Tokenization
    tokenized_dataset_test = dataset_test.map(fc.tokenizer_function, batched=True, fn_kwargs={'model_selected': model_selected, 'finetuned': False})
    test = tokenized_dataset_test
else:
    test = valid


unique_labels = processed_df[['original_labels', 'labels']].drop_duplicates()
sorted_labels = unique_labels.sort_values(by='labels')
original_labels = sorted_labels['original_labels'].tolist()

print("\n--------------------------------------------------------------------------")
print(f"Dataset(Train): {data_filename} {train_data} (row: {len(train)})")
print(f"Dataset(Valid): {data_filename_valid} {valid_data} (row: {len(valid)})") if train_only == 'n' else None
print(f"Dataset(Test): {data_filename_test} {test_data} (row: {len(test)})")
print(f"The number of label from training data: {count_label}")
print(f"Labels used: {original_labels}")
print("--------------------------------------------------------------------------\n")
# go_ahead = input("Is it correct? (y/n): ").strip().lower()
# if go_ahead != "y":
#     sys.exit()

# 3. Create model
model = fc.create_model(count_label, model_selected)

training_args = TrainingArguments(
    output_dir=result_dir,
    learning_rate=ln_rt,
    per_device_train_batch_size=bt_sz,
    per_device_eval_batch_size=bt_sz,
    num_train_epochs=epochs,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    eval_strategy="epoch" if train_only == 'n' else "no",
    logging_strategy="epoch",
    load_best_model_at_end=True if train_only == 'n' else False, # default: lowest validation loss
    dataloader_drop_last=False
)

tokenizer, _, data_collator = fc.setting_tokenizer(finetune_dir, model_selected)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid if train_only == 'n' else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=fc.compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    class_weights=class_weights_tensor # Giving more importance to underrepresented classes
)

# 4. Train the model
print("\n--------------------------------------------------------------------------")
if train_only == 'y':
    eval_info = k_fold_cross_validation(
        train=train,
        model=model,
        training_args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights_tensor=class_weights_tensor,
        k=k_fold_number
    )
    print(f"\nk_fold_results: \n{eval_info}")
else:
    trainer.train()
    eval_info = trainer.evaluate()
print("--------------------------------------------------------------------------\n")

# 5. Confusion matrix and Weighted average correctness When test data is labeled
if train_only == 'n':
    predictions = trainer.predict(valid)
    pred_labels = np.argmax(predictions.predictions, axis=-1)

    # 5-1. Confusion matrix
    conf_matrix = confusion_matrix(predictions.label_ids, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=original_labels, yticklabels=original_labels)

    plt.xticks(fontsize=9, rotation=45, ha='right')
    plt.yticks(fontsize=9)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix Heatmap - Fine-tuned {model_selected}")
    plt.tight_layout()

    savefig_name = "transformers/fig/" + model_selected + "_" + exp_name + ".png"
    plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
    # plt.show()

    # 5-2. Correctness
    df_correctness = fc.correctness(conf_matrix, original_labels)
    print("\n--------------------------------------------------------------------------")
    print(df_correctness)
    print("--------------------------------------------------------------------------\n")


# 6. Predict labels for test data
predictions_test = trainer.predict(test)
pred_labels_test = np.argmax(predictions_test.predictions, axis=-1)
id2label = {idx: label for idx, label in enumerate(original_labels)}
pred_text_labels = [id2label[label] for label in pred_labels_test]

if test_data_labeled == 'y':
    test_df = pd.DataFrame({"text": test["original_text"], "labels": test["original_labels"]})
else: 
    test_df = pd.DataFrame({"text": test["original_text"], "labels": [None] * len(test)})

column_name = "predicted_label_" + model_selected
test_df[column_name] = pred_text_labels

save_pred_filename = "transformers/result/predictions_" + model_selected + "_" + exp_name + ".xlsx"
pred_sheet_name = "Test_Data_Prediction"
test_df.to_excel(save_pred_filename, sheet_name=pred_sheet_name, index=True, engine="openpyxl")


# 7. Save the model
fc.save_finetuned(model, tokenizer, exp_name, model_selected)

end_time = datetime.now()
elapsed_time = end_time - start_time

print("\n-------------------------")
print(f"Start time: {start_time}")
print(f"End   time: {end_time}")
print(f"Execution Duration:     {elapsed_time}")
hours, remainder = divmod(elapsed_time.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
milliseconds = elapsed_time.microseconds / 1000
formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}.{int(milliseconds):03}"

# Delete result_dir once it finishes
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)

# 8. Save Results
new_row = {
    "Experiment Name": exp_name.upper(),
    "Model": model_selected,
    "Train Dataset": data_filename + " [" + " ".join(train_data) + "]",
    "Train Rows": len(train),
    "Validation Dataset": data_filename_valid + " [" + " ".join(valid_data) + "]" if train_only == 'n' else "",
    "Validation Rows": len(valid) if train_only == 'n' else "",
    "Test Dataset": data_filename_test + " [" + " ".join(test_data) + "]",
    "Test Rows": len(test),
    "Number of Labels from Training Data": count_label,
    "Labels Used": original_labels,
    "Learning Rate": ln_rt,
    "Batch Size": bt_sz,
    "Number of Epochs": epochs,
    "Execution Duration": formatted_time,
    "K-fold[k]": k_fold_number if train_only == 'y' else "",
    "Loss": eval_info['eval_loss'],
    "Accuracy": eval_info['eval_accuracy'],
    "Precision": eval_info['eval_precision'],
    "Recall": eval_info['eval_recall'],
    "F1-Score": eval_info['eval_f1']
}
fc.save_results(new_row)


# 9. Leave logs for each
file_name = "transformers/result/" + model_selected + "_" + exp_name + ".txt"
with open(file_name, "a") as file:
    file.write(f"- Dataset:\n")
    file.write(f"\tTrain data: {data_filename} {train_data} (row: {len(train)})\n")
    file.write(f"\tValid data: {data_filename_valid} {valid_data} (row: {len(valid)})\n") if train_only == 'n' else None
    file.write(f"\tTest data: {data_filename_test} {test_data} (row: {len(test)})\n")
    file.write(f"\tThe number of label from training data: {count_label}\n")
    file.write(f"\tLabels used: {original_labels}\n")

    file.write(f"- Learning rate: {ln_rt}\n")
    file.write(f"- Batch size: {bt_sz}\n")
    file.write(f"- Number of Epochs: {epochs}\n")
    file.write(f"- Execution Duration : {elapsed_time}\n")

    file.write(f"- Evaluation(K-fold[k={k_fold_number}]) :\n{eval_info}\n") if train_only == 'y' else None
    file.write(f"- Evaluation :\n{eval_info}\n") if train_only == 'n' else None
    file.write(f"\n# Weighted Averages of Error and Correctness of Fine-tuned {model_selected}\n") if train_only == 'n' else None
    file.write(df_correctness.to_string(index=False, header=True)) if train_only == 'n' else None
    file.write("\n------------------------------------------------------------\n")
