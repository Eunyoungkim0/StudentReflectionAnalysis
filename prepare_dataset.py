import pandas as pd
from openpyxl import load_workbook
import os
main_dir = "."
data_dir = os.path.join(main_dir, "Data")
data = os.path.join(data_dir, "Consensus_Data_Issue.xlsx")

save_file_path = os.path.join(data_dir, "Consensus_Data_Label_Count.xlsx")
save_file_path_binary = os.path.join(data_dir, "Consensus_Data_Label_Count_Binary.xlsx")
save_file_path_top8 = os.path.join(data_dir, "Consensus_Data_Top8_Challenges.xlsx")
save_file_path_one_label = os.path.join(data_dir, "Consensus_Data_One_Label_EK12.xlsx")

sheet_names = [
    'D-ESP4-1', 'D-ESU4-1',
    'D-ESP4-2', 'D-ESU4-2',
    'D-ESP4-3', 'D-ESU4-3',
    'D-ESP4-4', 'D-ESU4-4'
]

exclude_columns = ['TOTAL', 'labels', 'None', 'Other', 'Personal Issue', 'Assignments', 'Learning New Material']
label_count = pd.DataFrame(columns=["Ref_Num", "Sheet_Name", "Challenge", "Count"])
label_count_binary = pd.DataFrame(columns=["Ref_Num", "Sheet_Name", "Challenge", "Count"])

def data_preprocessing(data):
    df = pd.read_excel(data, sheet_name=sheet_name)
    df = df.drop_duplicates().reset_index(drop=True)

    df.columns = ["A", "B", "C", "D", "E", "F", "issue"]
    df['issue'] = df['issue'].fillna("None")
    df['issue'] = df['issue'].str.rstrip()
    df[['C', 'D', 'E', 'F']] = df[['C', 'D', 'E', 'F']].apply(lambda x: x.str.replace('\n', ' ', regex=True))
    df['text'] = df[['C', 'D', 'E', 'F']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    df = df.drop(columns=['B', 'C', 'D', 'E', 'F'])

    df_counts = df.groupby(['text', 'issue']).size().reset_index(name='count')
    df_pivot = df_counts.pivot(index='text', columns='issue', values='count').fillna(0).astype(int)

    return df_pivot

def calculate_total(df_pivot):
    df_pivot['TOTAL'] = df_pivot.loc[:, df_pivot.columns != 'text'].sum(axis=1)
    df_pivot.loc['TOTAL'] = df_pivot.sum(axis=0)

    return df_pivot


def create_label_count(df, isBinary=False):
    global label_count, label_count_binary

    existing_columns_to_drop = [col for col in exclude_columns if col in df.columns]
    columns_sum = df.drop(columns=existing_columns_to_drop).sum().index

    for column in columns_sum:
        total_count = df.loc['TOTAL', column]
        reflection_num = sheet_name[-1]
        new_row = pd.DataFrame({
            "Ref_Num": [reflection_num],
            "Sheet_Name": [sheet_name],
            "Challenge": [column],
            "Count": [total_count]
        })

        if(isBinary):
          label_count_binary = pd.concat([label_count_binary, new_row], ignore_index=True)
        else:
          label_count = pd.concat([label_count, new_row], ignore_index=True)

    return df

def assign_label(row):
    if row.max() == 0:
        return "Other"

    sorted_values = row.sort_values(ascending=False)

    for value in sorted_values.unique():
        max_columns = sorted_values[sorted_values == value].index.tolist()
        temp = {}

        for column in max_columns:
            if column in ref_top_challenges['Challenge'].values:
                index = ref_top_challenges[ref_top_challenges['Challenge'] == column].index[0]
                temp[column] = index

        if temp:
            return min(temp, key=temp.get)

    return "Other"

# Create Consensus_Data_Label_Count.xlsx
with pd.ExcelWriter(save_file_path, mode="w", engine="openpyxl") as writer:
    for sheet_name in sheet_names:
        df_pivot = data_preprocessing(data)
        df_pivot = calculate_total(df_pivot)
        df_pivot = create_label_count(df_pivot, False)

        df_pivot.to_excel(writer, sheet_name=sheet_name, index=True)


# Create Consensus_Data_Label_Count_Binary.xlsx
with pd.ExcelWriter(save_file_path_binary, mode="w", engine="openpyxl") as binary_writer:
    for sheet_name in sheet_names:
        df_pivot_binary = data_preprocessing(data)
        df_pivot_binary = df_pivot_binary.apply(lambda x: x.map(lambda y: 1 if y > 0 else 0))
        df_pivot_binary = calculate_total(df_pivot_binary)
        df_pivot_binary = create_label_count(df_pivot_binary, True)

        df_pivot_binary.to_excel(binary_writer, sheet_name=sheet_name, index=True)


label_count_by_reflection = label_count.groupby(['Ref_Num', 'Challenge'], as_index=False)['Count'].sum()
label_count_by_reflection = label_count_by_reflection.sort_values(by=['Ref_Num', 'Count'], ascending=[True, False])
label_count_by_reflection = label_count_by_reflection.reset_index(drop=True)

label_count_overall = label_count.groupby(['Challenge'], as_index=False)['Count'].sum()
label_count_overall = label_count_overall.sort_values(by='Count', ascending=False)
label_count_overall = label_count_overall.reset_index(drop=True)

label_count_by_reflection['Count'] = pd.to_numeric(label_count_by_reflection['Count'], errors='coerce')
label_count_overall['Count'] = pd.to_numeric(label_count_overall['Count'], errors='coerce')

# Top # Challenges Overall
# top_8_challenges = label_count_overall.head(8)
top_6_challenges = label_count_overall.head(6)

# Top 8 Challenges by reflection
top_8_challenges_by_reflection = label_count_by_reflection.groupby('Ref_Num').apply(
    lambda x: x[['Ref_Num', 'Challenge', 'Count']].nlargest(8, 'Count')).reset_index(drop=True)


# Create Consensus_Data_Top8_Challenges.xlsx
with pd.ExcelWriter(save_file_path_top8, engine="openpyxl") as writer:
    for ref_num, group in top_8_challenges_by_reflection.groupby('Ref_Num'):
        group[['Challenge', 'Count']].to_excel(writer, sheet_name=str(ref_num), index=False)

# Create Consensus_Data_One_Label.xlsx
with pd.ExcelWriter(save_file_path_one_label, mode="w", engine="openpyxl") as writer:
    for sheet_name in sheet_names:
        df_pivot = data_preprocessing(data)

        existing_columns_to_drop = [col for col in exclude_columns if col in df_pivot.columns]
        relevant_columns = df_pivot.drop(columns=existing_columns_to_drop)

        reflection_num = sheet_name[-1]
        # ref_top_challenges = top_8_challenges_by_reflection[top_8_challenges_by_reflection['Ref_Num'] == reflection_num]
        ref_top_challenges = top_6_challenges

        challenge_columns = ref_top_challenges['Challenge'].values
        relevant_columns = relevant_columns.loc[:, relevant_columns.columns.isin(challenge_columns)]

        df_pivot['labels'] = relevant_columns.apply(assign_label, axis=1)

        df_pivot.to_excel(writer, sheet_name=sheet_name, index=True)