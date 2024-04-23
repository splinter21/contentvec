import os
import librosa
import soundfile as sf
import argparse
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_batch(file_chunk, in_dir, out_dir, target_sr):
    for filename in tqdm(file_chunk):
        try:
            audio, sr = librosa.load(filename, sr=None)

            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            out_audio = os.path.join(out_dir, os.path.relpath(filename, in_dir)).rsplit('.', 1)[0] + '.wav'
            os.makedirs(os.path.dirname(out_audio), exist_ok=True)
            
            sf.write(out_audio, audio, target_sr, format='WAV')

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            with open('error.log', 'a', encoding='utf-8') as error_log:
                error_log.write(f'Error: {filename}\n')

def parallel_process(filelist, num_processes, in_dir, out_dir, target_sr):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [executor.submit(process_batch, filelist[rank::num_processes], in_dir, out_dir, target_sr) for rank in range(num_processes)]
        for task in tasks:
            task.result()

def get_filelist(in_dir):
    extensions = ['wav', 'ogg', 'opus', 'snd', 'flac']
    files = []
    for ext in extensions:
        files.extend(glob(f"{in_dir}/**/*.{ext}", recursive=True))
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=r"dataset_raw")
    parser.add_argument("--out_dir", type=str, default=r"dataset_raw")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument('--num_processes', type=int, default=20)
    args = parser.parse_args()

    print('Loading files...')
    filelist = get_filelist(args.in_dir)
    print(f'Number of files: {len(filelist)}')
    print('Start Resample...')

    parallel_process(filelist, args.num_processes, args.in_dir, args.out_dir, args.target_sr)
