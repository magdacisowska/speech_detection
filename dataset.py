from urllib.request import urlretrieve
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def download_dataset(filename):
    """Downloads AVA-Speech dataset"""
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            url = "https://s3.amazonaws.com/ava-dataset/trainval/" + line
            dst = "dataset/" + line[:len(line) - 1]
            urlretrieve(url, dst)
            print("Downloaded file: " + line)


def cut_out_dataset(filename):
    """Cuts out 15 useful minutes"""
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            try:
                ffmpeg_extract_subclip(
                        'dataset/' + line[:len(line) - 1],
                        900,
                        1800,
                        'cut_dataset/' + line[:len(line) - 1]
                    )
            except IOError:
                pass


def audio_from_video(filename):
    """Extracts audio from cut video"""
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            try:
                command = "ffmpeg -i cut_dataset/" + line[:len(line) - 1] +\
                          " -ab 160k -ac 2 -ar 22000 -vn audio_dataset/" + line[:len(line) - 4] + "wav"
                subprocess.call(command, shell=False)
            except IOError:
                pass


# download_dataset('test_labels')
# cut_out_dataset('test_labels')
# audio_from_video('test_labels')
