import os
import librosa
import soundfile as sf
import argparse
import torch
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from multiprocessing import Manager
from torchaudio.transforms import Resample

def process_batch(file_chunk, in_dir, out_dir, target_sr, log_queue):
    RESAMPLE_KERNEL = {}
    for filename in tqdm(file_chunk):
        try:
            audio, sr = librosa.load(filename, sr=None, mono=True)  # 加上 mono=True 参数以加载为单声道
            duration = librosa.get_duration(y=audio, sr=sr)  # 计算音频时长

            if duration > 30 or duration < 1:
                print(f"\nSkip: {filename} - Duration: {duration:.2f}s")
                log_queue.put(f'Skip: {filename}\n')
                continue

            if sr != target_sr:
                if sr not in RESAMPLE_KERNEL:
                    RESAMPLE_KERNEL[sr] = Resample(sr, target_sr).to('cuda')
                audio_torch = RESAMPLE_KERNEL[sr](torch.FloatTensor(audio).unsqueeze(0).to('cuda'))
                audio = audio_torch.squeeze().cpu().numpy()

            out_audio = os.path.join(out_dir, os.path.relpath(filename, in_dir)).rsplit('.', 1)[0] + '.wav'
            os.makedirs(os.path.dirname(out_audio), exist_ok=True)

            sf.write(out_audio, audio, target_sr, format='WAV')

        except Exception as e:
            print(f"\nError: {filename}: {e}")
            log_queue.put(f'Error: {filename}\n')

def log_writer(log_queue, log_file_path):
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        while True:
            message = log_queue.get()
            if message == 'STOP':
                break
            log_file.write(message)

def parallel_process(filelist, num_processes, in_dir, out_dir, target_sr):
    manager = Manager()
    log_queue = manager.Queue()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        log_writer_future = executor.submit(log_writer, log_queue, 'error.log')
        tasks = [executor.submit(process_batch, filelist[rank::num_processes], in_dir, out_dir, target_sr, log_queue) for rank in range(num_processes)]
        for task in tasks:
            task.result()

        log_queue.put('STOP')
        log_writer_future.result()

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
    parser.add_argument('--num_processes', type=int, default=16)
    args = parser.parse_args()

    print('Loading files...')
    filelist = get_filelist(args.in_dir)
    print(f'Number of files: {len(filelist)}')
    print('Start Resample...')

    parallel_process(filelist, args.num_processes, args.in_dir, args.out_dir, args.target_sr)
