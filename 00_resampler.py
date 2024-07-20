import os
import librosa
import soundfile as sf
import argparse
import torch
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
from multiprocessing import Manager
from torchaudio.transforms import Resample

def process_batch(rank, filelist, in_dir, out_dir, target_sr, log_queue, num_gpus):
    torch.cuda.set_device(rank % num_gpus)  # Set the device to the corresponding GPU
    RESAMPLE_KERNEL = {}
    for filename in tqdm(filelist[rank::num_gpus]):
        try:
            audio, sr = librosa.load(filename, sr=None, mono=True)
            duration = librosa.get_duration(y=audio, sr=sr)

            if duration > 30 or duration < 1:
                print(f"\nSkip: {filename} - Duration: {duration:.2f}s")
                log_queue.put(f'Skip: {filename}\n')
                continue

            if sr != target_sr:
                if sr not in RESAMPLE_KERNEL:
                    RESAMPLE_KERNEL[sr] = Resample(sr, target_sr).to(rank % num_gpus)
                audio_torch = RESAMPLE_KERNEL[sr](torch.FloatTensor(audio).unsqueeze(0).to(rank % num_gpus))
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

def parallel_process(filelist, num_gpus, in_dir, out_dir, target_sr):
    manager = Manager()
    log_queue = manager.Queue()

    log_writer_process = mp.Process(target=log_writer, args=(log_queue, 'error.log'))
    log_writer_process.start()

    processes = []
    for rank in range(num_gpus):
        p = mp.Process(target=process_batch, args=(rank, filelist, in_dir, out_dir, target_sr, log_queue, num_gpus))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    log_queue.put('STOP')
    log_writer_process.join()

def get_filelist(in_dir):
    extensions = ['wav']
    files = []
    for ext in extensions:
        files.extend(glob(f"{in_dir}/**/*.{ext}", recursive=True))
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=r"/home/m123/WenetSpeech4TTS_123")
    parser.add_argument("--out_dir", type=str, default=r"dataset_raw")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument('--num_gpus', type=int, default=5)
    args = parser.parse_args()

    print('Loading files...')
    filelist = get_filelist(args.in_dir)
    print(f'Number of files: {len(filelist)}')
    print('Start Resample...')

    parallel_process(filelist, args.num_gpus, args.in_dir, args.out_dir, args.target_sr)