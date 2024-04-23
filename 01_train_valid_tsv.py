import os
import random
import soundfile as sf
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def main(filelist):
    audio_dict = {}
    for file in tqdm(filelist):
        with sf.SoundFile(file) as audio:
            total_frames = len(audio)
            audio_dict[file] = total_frames
    return audio_dict

def merge_dicts(dicts):
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    return merged_dict

if __name__ == '__main__':
    root_dir = r'/home/bfloat16/contentvec/dataset_raw'
    num_processes = 10

    audio_files = []
    extensions = ["wav", "mp3", "ogg", "flac", "opus", "snd"]
    for ext in extensions:
        audio_files.extend(glob(os.path.join(root_dir, f'**/*.{ext}'), recursive=True))

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = []
        tasks = [executor.submit(main, audio_files[int(i * len(audio_files) / num_processes): int((i + 1) * len(audio_files) / num_processes)]) for i in range(num_processes)]
        for future in tasks:
            results.append(future.result())

    audio_dict = merge_dicts(results)

    validation_set = random.sample(audio_files, 50)
    training_set = [file for file in audio_files if file not in validation_set]

    os.makedirs('./data/metadata', exist_ok=True)
    
    with open('./data/metadata/train.tsv', 'w', encoding='utf-8') as train_file:
        train_file.write(f'{root_dir}\n')
        for audio_path in training_set:
            relative_path = os.path.relpath(audio_path, root_dir)
            total_frames = audio_dict[audio_path]
            train_file.write(f'{relative_path}\t{total_frames}\n')

    with open('./data/metadata/valid.tsv', 'w', encoding='utf-8') as val_file:
        val_file.write(f'{root_dir}\n')
        for audio_path in validation_set:
            relative_path = os.path.relpath(audio_path, root_dir)
            total_frames = audio_dict[audio_path]
            val_file.write(f'{relative_path}\t{total_frames}\n')