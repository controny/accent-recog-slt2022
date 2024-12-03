import pandas as pd
import os
import numpy as np
import subprocess
from pydub import AudioSegment
from tqdm import tqdm

# Load the dataset
dataset_dir = 'data/speech_accent'
file_path = f'{dataset_dir}/speakers_all.csv'
data = pd.read_csv(file_path)

# Filter out countries with more than 10 samples
more_than_10 = data['country'].value_counts()
countries_filtered = more_than_10[more_than_10 >= 10].index
data_filtered = data[data['country'].isin(countries_filtered)]

# Define the directory containing the recordings and the directory for truncated files
recordings_dir = f'{dataset_dir}/recordings/recordings'
truncated_dir = f'{dataset_dir}/truncated_recordings'
os.makedirs(truncated_dir, exist_ok=True)

# Initialize lists to store training and testing data
train_data = []
test_data = []

# Split the data for each native language into train and test sets (90% train, 10% test)
np.random.seed(42)  # For reproducibility
for country in countries_filtered:
    language_data = data_filtered[data_filtered['country'] == country]
    language_samples = language_data.sample(frac=1, random_state=42)  # Shuffle the samples

    # Split into 90% train and 10% test
    split_idx = int(len(language_samples) * 0.9)
    train_samples = language_samples[:split_idx]
    test_samples = language_samples[split_idx:]

    # Add to the respective lists
    train_data.append(train_samples)
    test_data.append(test_samples)

# Concatenate all train and test samples
train_df = pd.concat(train_data).reset_index(drop=True)
test_df = pd.concat(test_data).reset_index(drop=True)
print('train size', len(train_df))
print('test size', len(test_df))

# Prepare the final train and test DataFrames with the required format
def truncate_audio(file_path, output_path, max_duration=10):
    try:
        if os.path.exists(output_path):
            return
        audio = AudioSegment.from_mp3(file_path)
        truncated_audio = audio[:max_duration * 1000]  # Truncate to max_duration seconds
        truncated_audio.export(output_path, format="mp3")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

train_final = []
test_final = []

for df, final_list in zip([train_df, test_df], [train_final, test_final]):
    for idx, row in tqdm(df.iterrows()):
        file_name = row['filename'] + '.mp3'
        original_file = os.path.join(recordings_dir, file_name)
        truncated_file = os.path.join(truncated_dir, file_name)
        if not os.path.exists(original_file):
            continue
        truncate_audio(original_file, truncated_file)

        final_list.append({
            'ID': idx,
            'utt_id': row['filename'],
            'wav': truncated_file,
            'wav_format': 'mp3',
            'text': '',
            'duration': 10,
            'accent': row['country']
        })

train_final_df = pd.DataFrame(train_final)
test_final_df = pd.DataFrame(test_final)

# Save the train and test splits to CSV files
train_final_df.to_csv(f'{dataset_dir}/train.csv', index=False)
test_final_df.to_csv(f'{dataset_dir}/test.csv', index=False)
