import pandas as pd
import numpy as np
import glob
import os

save_path = f"/home/petr/Downloads/09_09_results"
skills = ["peg_pick404", "peg_door404", "peg_place404", "probe_pick404", "slider_move404"]
sessions = ["manipulation_demo404_augment_24_session",
            "manipulation_demo404_augment_32_session",
            "manipulation_demo404_augment_48_session",
            "manipulation_demo404_augment_session",
]

def swap_train_test(s):
    if s == "Train": return "Test"
    elif s == "Test": return "Train"
    else: raise Exception()

for subset in ['Train', 'Test']:
    rows = []
    columns = []
    for session in sessions:

        for skill in skills:

            # Define the directory containing the CSV files
            directory_path = f'/{save_path}/{session}/{skill}'

            # Pattern to match files starting with "Test_dataset_"
            if session == "manipulation_demo404_augment_session":
                file_pattern = os.path.join(directory_path, f'{swap_train_test(subset)}_dataset_*.csv')
            else:
                file_pattern = os.path.join(directory_path, f'{subset}_dataset_*.csv')


            # Use glob to find all files matching the pattern
            csv_files = glob.glob(file_pattern)

            # List to store data from all CSV files


            # Load each CSV file and append its data to the list
            for i,file in enumerate(csv_files):
                # Read the CSV file
                data = pd.read_csv(file)
                
                if i == 0: # construct the header once
                    columns = list(data.columns)
                    columns.pop(0) # drop index column
                    columns.insert(0, "Session")
                    columns.insert(0, "Skill")
                    columns.insert(0, "Model")

                row = list(data.to_numpy().squeeze())
                row.pop(0) # drop index column
                row.insert(0, str(file).split("/")[-3]) # insert session
                row.insert(0, str(file).split("/")[-2]) # insert skill
                row.insert(0, str(file).split("/")[-1]) # insert model
                rows.append(row)

    print(columns)
    for row in rows:
        print(row)

    df = pd.DataFrame(rows, columns=columns)
    df_sorted = df.sort_values(by=['Skill', 'Accuracy'], ascending=[True, False])
    # df_sorted.to_csv(f"{save_path}/{subset}_results.csv")

    average_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()
    print(average_accuracy)
    average_accuracy = average_accuracy.sort_values(by=['Accuracy'], ascending=[False])
    average_accuracy.to_csv(f"{save_path}/{subset}_sortedmodel_results.csv")