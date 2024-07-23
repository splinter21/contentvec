import argparse
from glob import glob
from mutagen.wave import WAVE
from mutagen.oggvorbis import OggVorbis
from mutagen.oggopus import OggOpus
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import librosa

rich_progress = Progress(TextColumn("Running: "), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeElapsedColumn(), "|", TimeRemainingColumn())

def sec_to_time(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return int(hours), int(minutes), seconds

def calculate_durations(filenames, ca_fast=False):
    total_duration_sec = 0
    max_duration_sec = 0
    min_duration_sec = float('inf')
    
    with Progress() as rich_progress:
        task2 = rich_progress.add_task("Processing", total=len(filenames))
        
        for filename in filenames:
            try:
                if ca_fast:
                    if filename.endswith('.wav'):
                        audio = WAVE(filename)
                    elif filename.endswith('.ogg'):
                        audio = OggVorbis(filename)
                    elif filename.endswith('.opus'):
                        audio = OggOpus(filename)
                    else:
                        raise ValueError(f"Unsupported file format: {filename}")
                    
                    file_duration_sec = audio.info.length  # Duration in seconds
                else:
                    audio, sr = librosa.load(filename, sr=None)
                    file_duration_sec = librosa.get_duration(y=audio, sr=sr)
                
                total_duration_sec += file_duration_sec
                max_duration_sec = max(max_duration_sec, file_duration_sec)
                min_duration_sec = min(min_duration_sec, file_duration_sec)
                rich_progress.update(task2, advance=1)
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return total_duration_sec, max_duration_sec, min_duration_sec

def parallel_process(filenames, num_processes, ca_fast):
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [executor.submit(calculate_durations, filenames[int(i * len(filenames) / num_processes): int((i + 1) * len(filenames) / num_processes)], ca_fast) for i in range(num_processes)]
        for future in tasks:
            results.append(future.result())
    return results

def aggregate_results(results):
    total_duration_sec = sum(result[0] for result in results)
    max_duration_sec = max(result[1] for result in results)
    min_duration_sec = min(result[2] for result in results if result[2] != float('inf'))  # Avoiding inf if possible
    return sec_to_time(total_duration_sec), sec_to_time(max_duration_sec), sec_to_time(min_duration_sec)

def main(in_dir, num_processes, ca_fast):
    print('Loading audio files...')
    extensions = ["wav", "mp3", "ogg", "flac", "opus", "snd"]
    filenames = []
    for ext in extensions:
        filenames.extend(glob(f"{in_dir}/**/*.{ext}", recursive=True))
    print("==========================================================================")
    
    if filenames:
        results = parallel_process(filenames, num_processes, ca_fast)
        total_duration, max_duration, min_duration = aggregate_results(results)
        print("==========================================================================")
        print(f"SUM: {len(filenames)} files\n")
        print(f"SUM: {total_duration[0]:02d}:{total_duration[1]:02d}:{total_duration[2]:05.2f}")
        print(f"MAX: {max_duration[0]:02d}:{max_duration[1]:02d}:{max_duration[2]:05.2f}")
        print(f"MIN: {min_duration[0]:02d}:{min_duration[1]:02d}:{min_duration[2]:05.2f}")
        return(f'{total_duration[0]:02d}:{total_duration[1]:02d}:{total_duration[2]:05.2f}')

    else:
        print("No audio files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=r"dataset_raw")
    parser.add_argument('--num_processes', type=int, default=50)
    parser.add_argument('--ca_fast', action='store_true', help='Use fast calculation method with mutagen')
    args = parser.parse_args()

    main(args.in_dir, args.num_processes, args.ca_fast)