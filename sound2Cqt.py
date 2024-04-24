import os
import librosa
import librosa.display
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def audio_to_time_hz_spectrogram(input_folder, output_folder, sr=43900, window_size=0.182, output_size=(224, 224)):
    # 计算帧数
    n_fft = int(window_size * sr)

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

                # 读取音频文件并转换为时频图
                y, sr = librosa.load(input_file, sr=sr)
                S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4))

                # 显示时频图并保存为图片文件
                librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, hop_length=n_fft // 4,
                                          x_axis='time', y_axis='cqt_hz',cmap='viridis')
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
output_folder = 'CQT_spectrograms'

# 将音频文件转换为时频图
audio_to_time_hz_spectrogram(input_folder, output_folder)


