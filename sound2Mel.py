import os
import librosa
import librosa.display
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def audio_to_mel_spectrogram(input_folder, output_folder, sr=43900, n_mels=64, hop_length=64, window_length=1024, output_size=(224, 224)):
    # 遍历输入文件夹中的每个生物文件夹
    for root, dirs, files in os.walk(input_folder):
        for folder in dirs:
            input_subfolder = os.path.join(root, folder)
            output_subfolder = os.path.join(output_folder, folder)
            os.makedirs(output_subfolder, exist_ok=True)

            # 遍历生物文件夹中的音频文件
            for file in os.listdir(input_subfolder):
                input_file = os.path.join(input_subfolder, file)
                output_file = os.path.join(output_subfolder, file.replace('.wav', '.png'))

                # 读取音频文件并转换为梅尔频谱图
                y, sr = librosa.load(input_file, sr=sr)
                S = librosa.feature.melspectrogram(y, sr=sr, n_fft=window_length, hop_length=hop_length, n_mels=n_mels)

                # 显示梅尔频谱图并保存为图片文件
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=hop_length,
                                          x_axis=None, y_axis=None, cmap='viridis')
                plt.gcf().set_size_inches(output_size[0] / 100, output_size[1] / 100)
                plt.gca().axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()

                # 调整图片大小
                img = Image.open(output_file)
                img = img.resize(output_size, Image.ANTIALIAS)
                img.save(output_file)

# 设置输入和输出文件夹
input_folder = 'processed_archive'
output_folder = 'mel_spectrograms'

# 将音频文件转换为梅尔频谱图
audio_to_mel_spectrogram(input_folder, output_folder)

