import argparse
from resemblyzer import VoiceEncoder, preprocess_wav
import torch
import random
from os.path import join, exists
from tqdm import tqdm
import pickle
import parselmouth
from concurrent.futures import ProcessPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_f0_with_parselmouth(filepath):
    snd = parselmouth.Sound(filepath)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    return pitch_values.mean(), pitch_values.max(), pitch_values.min()

def extract_embedding(filepath, encoder):
    '''
    Embeddings from: Generalized End-To-End Loss for Speaker Verification 
    '''
    wav = preprocess_wav(filepath)
    file_embedding = encoder.embed_utterance(wav)
    embedding = torch.tensor(file_embedding)
    return embedding

def calculate_average(speaker_dict):
    if 'train' not in speaker_dict:
        return {}

    speaker_avg_dict = {'train': {}}

    for speaker_id in speaker_dict['train']:
        samples = speaker_dict['train'][speaker_id]

        emb_avg = torch.zeros(256)
        f0_mean_avg = 0
        f0_max_avg = 0
        f0_min_avg = 0

        total_samples = len(samples)

        for embedding, (f0_mean, f0_max, f0_min) in samples:
            emb_avg += embedding
            f0_mean_avg += f0_mean
            f0_max_avg += f0_max
            f0_min_avg += f0_min

        emb_avg /= total_samples
        f0_mean_avg /= total_samples
        f0_max_avg /= total_samples
        f0_min_avg /= total_samples

        speaker_avg_dict['train'][speaker_id] = \
            emb_avg.tolist(), (f0_mean_avg, f0_max_avg, f0_min_avg)
        
    return speaker_avg_dict 

def process_files(filelist, root_folder):
    encoder = VoiceEncoder()
    speaker_dict = {}
    for filepath in tqdm(filelist):        
        speaker_id = str(filepath)
        filepath = join(root_folder, filepath)
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue

        embedding = extract_embedding(filepath, encoder=encoder)
        f0_mean, f0_max, f0_min = get_f0_with_parselmouth(filepath)
        f0_mean, f0_max, f0_avg = int(f0_mean), int(f0_max), ((f0_mean + f0_max) / 2)
        speaker_dict[speaker_id] = embedding.numpy(), (f0_mean, f0_max, f0_avg)
    return speaker_dict

def parallel_process(filenames, root_folder, num_processes):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        chunk_size = len(filenames) // num_processes
        for i in range(num_processes):
            start = i * chunk_size
            end = None if i == num_processes - 1 else (i + 1) * chunk_size
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_files, file_chunk, root_folder))
        
        speaker_dict = {}
        for task in tqdm(tasks, position=0):
            result = task.result()
            speaker_dict.update(result)
    return speaker_dict

def generate_list_dict_from_list(filelist_train, filelist_val, root_folder, output_filepath, num_processes):
    speaker_dict = {'train': {}, 'valid': {}}

    # Process validation files in parallel
    speaker_dict['valid'] = parallel_process(filelist_val, root_folder, num_processes)

    # Process training files in parallel
    speaker_dict['train'] = parallel_process(filelist_train, root_folder, num_processes)

    return speaker_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input_train', default="data/metadata/train.tsv")
    parser.add_argument('-v', '--input_val', default="data/metadata/valid.tsv")
    parser.add_argument('-d', '--dataset_dir', default="dataset_raw", help='Dataset root folder')
    parser.add_argument('-o', '--output', default='data/spk2info.dict', help='Output folder')
    parser.add_argument('-n', '--num_process', type=int, default=6, help='Number of processes for multiprocessing')
    args = parser.parse_args()

    with open(args.input_train, "r", encoding='utf-8') as file:
        data = file.readlines()[1:]
    filelist_train = [line.split("\t")[0] for line in data]

    with open(args.input_val, "r", encoding='utf-8') as file:
        data = file.readlines()[1:]
    filelist_val = [line.split("\t")[0] for line in data]

    speaker_list_dict = generate_list_dict_from_list(filelist_train, filelist_val, args.dataset_dir, args.output, args.num_process)

    del filelist_train
    del filelist_val

    with open(args.output, 'wb') as file:
        pickle.dump(speaker_list_dict, file)

if __name__ == "__main__":
    main()
