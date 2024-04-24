import os
import librosa
import numpy as np
import soundfile as sf

def process_audio(input_folder, output_folder, target_length=8000):
    # 遍历输入文件夹中的每个生物文件夹
    for root, dirs, files in os.walk(input_folder):
        for folder in dirs:
            input_subfolder = os.path.join(root, folder)
            output_subfolder = os.path.join(output_folder, folder)
            os.makedirs(output_subfolder, exist_ok=True)

            # 遍历生物文件夹中的音频文件
            for file in os.listdir(input_subfolder):
                input_file = os.path.join(input_subfolder, file)
                output_file = os.path.join(output_subfolder, file)

                # 裁剪音频文件
                y, sr = librosa.load(input_file, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)

                if duration > target_length / 1000:  # 如果音频长度大于目标长度（秒）
                    clip_length = duration - target_length / 1000
                    left = duration - target_length / 1000 // 2
                    start_time = int(left * sr)
                    end_time = int(start_time + target_length/1000 * sr)
                    print(start_time/sr,end_time/sr)
                    y = y[start_time:end_time]
                else:
                    pad_length = int((target_length / 1000) - duration)
                    print("pad_length:", pad_length)
                    #left_pad = int(pad_length // 2)  # 左侧填充宽度
                    left_pad = 0  # 左侧填充宽度
                    #right_pad = int(pad_length - left_pad)  # 右侧填充宽度
                    right_pad = 0  # 右侧填充宽度
                    print("Left pad:", left_pad)
                    print("Right pad:", right_pad)
                    y = np.pad(y, (left_pad * sr, right_pad * sr), 'constant')

                # 保存处理后的音频文件
                sf.write(output_file, y, sr)


# 设置输入和输出文件夹
input_folder = 'archive'
output_folder = 'processed_archive'

# 处理音频文件
process_audio(input_folder, output_folder)


