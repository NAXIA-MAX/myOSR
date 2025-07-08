import numpy as np
import os
import io
import sys
import signal as signalprocess
import matplotlib.pyplot as plt
import threading
from datetime import datetime
from scipy import signal
from skimage.transform import resize as skimage_resize
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure
"""
使用多线程批量生成图片数组
"""

# 创建一个全局 Event 作为线程终止信号
stop_event = threading.Event()
#信号捕捉
def signal_handler(sig, frame):
    print("\n捕获到中断信号，退出子线程...")
    stop_event.set()  # 设置退出标志
# 注册信号处理函数
signalprocess.signal(signalprocess.SIGINT, signal_handler)

class settings:
    def __init__(self,IF=4.092e6,samplingFreq=16.368e6,codeFreqBasis=1.023e6,codeLength= 1023):
        self.IF=IF
        self.samplingFreq=samplingFreq
        self.codeFreqBasis=codeFreqBasis
        self.codeLength=codeLength

def recurrenceImage(signal, dimension=2, delay=1, epsilon=None, normalize=True):
    """
    生成递归图（Recurrence Plot）
    :param signal: 一维信号数据
    :param dimension: 相空间嵌入维度
    :param delay: 延迟时间
    :param epsilon: 相似性阈值，若为 None 则使用自动比例
    :param normalize: 是否对信号归一化至 [0, 1]
    :return: 递归图矩阵
    """
    signal=signal[::500]#下采样处理
    N = len(signal) - (dimension - 1) * delay
    embedded = np.array([signal[i:i + dimension * delay:delay] for i in range(N)])

    if normalize:
        scaler = MinMaxScaler()
        embedded = scaler.fit_transform(embedded)

    distance_matrix = np.linalg.norm(embedded[:, None] - embedded[None, :], axis=2)

    if epsilon is None:
        epsilon = 0.1 * np.max(distance_matrix)  # 设定相似性阈值

    RP = (distance_matrix < epsilon).astype(float)
    RP=1-RP
    RP=np.flipud(RP)
    return RP

def histogram(data, numofbins=4):
    hist, _ = np.histogram(data, bins=numofbins)
    height = np.max(hist) if hist.size > 0 else 0
    hist_image = np.full((height, numofbins), 255, dtype=np.uint8)
    # 填充直方图数据
    for i, h_val in enumerate(hist):
        if h_val > 0:
            hist_image[height - h_val : height, i] = 0
    return hist_image
    

def welch_psd_to_image_array(frequencies, psd_db, image_height=256, bg_color=255, line_color=0):
    """
    辅助函数：将频率和PSD(dB)数据直接转换为二维NumPy数组（灰度图像）。
    背景默认为白色(255)，线条默认为黑色(0)。
    """
    # 如果频率或PSD数据为空，则返回一个具有指定高度和0宽度的背景色图像
    if frequencies.size == 0 or psd_db.size == 0:
        return np.full((image_height, 0), bg_color, dtype=np.uint8)

    image_width = len(frequencies)
    # 初始化图像数组为背景色
    image_array = np.full((image_height, image_width), bg_color, dtype=np.uint8)

    # 将psd_db值归一化以适应图像高度
    min_db = np.min(psd_db)
    max_db = np.max(psd_db)

    # 处理所有psd_db值相同（即PSD为平线）的特殊情况
    if max_db == min_db:
        # 在这种情况下，可以将线绘制在图像的中间，或底部/顶部
        # 此处示例为绘制在图像高度的一半位置
        scaled_heights = np.full_like(psd_db, image_height // 2, dtype=int)
    else:
        # 正常归一化：首先将值域转换到 [0, 1] 范围
        normalized_psd = (psd_db - min_db) / (max_db - min_db)
        # 然后缩放到 [0, image_height - 1] 范围，并转换为整数作为像素行索引
        scaled_heights = (normalized_psd * (image_height - 1)).astype(int)

    # 填充图像数组，为每个频率点（图像的每一列）绘制PSD值
    for i in range(image_width):
        h = scaled_heights[i]
        # 确保计算出的高度h在有效像素范围内 [0, image_height - 1]
        h = np.clip(h, 0, image_height - 1)
        # 图像填充
        start_row = image_height - 1 - h
        image_array[start_row : image_height, i] = line_color
        # row_index = image_height - 1 - h
        # image_array[row_index, i] = line_color

    return image_array


def getWelchImage(data, s_sampling_freq_hz, Nfft=2048, window_type='hann', nperseg=2048, noverlap=1024, image_height=256):
    """
    生成Welch PSD的灰度图像二维数组。
    """
    # 使用Welch方法计算功率谱密度
    frequencies, psd = signal.welch(
        data,
        fs=s_sampling_freq_hz,    # 采样频率 (Hz)
        window=window_type,       # 窗函数类型
        nperseg=nperseg,          # 每个数据段的长度
        noverlap=noverlap,        # 数据段之间的重叠点数
        nfft=Nfft,                # FFT点数
        return_onesided=True,     # 对于实数输入，通常返回单边谱
        scaling='density'         # 'density' 输出功率谱密度 (例如 V**2/Hz)
    )

    epsilon = 1e-12 # 防止对0或负数取对数
    # 在计算对数之前，确保PSD值为正
    psd_positive = np.maximum(psd, epsilon)
    psd_db = 10 * np.log10(psd_positive) # 将PSD值转换为dB单位

    # 调用辅助函数，将计算得到的频率和PSD(dB)数据转换为图像数组
    # 默认生成白底黑线的图像
    fft_image = welch_psd_to_image_array(frequencies, psd_db, image_height=image_height)
    
    return fft_image

#4类图像预处理
def imageProcess(data_array,s,target_shape,channel):
     #做STFT变换
    _,_,Z=signal.stft(data_array,s.samplingFreq/1e6,window='hann',nperseg=2048,noverlap=1024)
    real_Zxx = 10 * np.log10(np.abs(Z))
    #获取频谱
    fftImage=getWelchImage(data_array,s.samplingFreq/1e6)
    #获得时域分布柱状图
    time_histogram=histogram(data_array)
    #绘制递归图
    reImage=recurrenceImage(data_array)
    #resize为target_shape
    reImage_shape=skimage_resize(reImage, target_shape, order=3, anti_aliasing=True, preserve_range=True)
    timeHistogram_shape=skimage_resize(time_histogram, target_shape, order=3, anti_aliasing=True, preserve_range=True)
    fftImage_shape=skimage_resize(fftImage, target_shape, order=3, anti_aliasing=True, preserve_range=True)
    I_skimage = skimage_resize(real_Zxx, target_shape, order=3, anti_aliasing=True, preserve_range=True)
    #归一化到0-255
    reImage_uint8=exposure.rescale_intensity(reImage_shape, out_range=(0, 255)).astype(np.uint8)
    timeHistogram_uint8=exposure.rescale_intensity(timeHistogram_shape, out_range=(0, 255)).astype(np.uint8)
    fftImage_uint8=exposure.rescale_intensity(fftImage_shape, out_range=(0, 255)).astype(np.uint8)
    I_skimage_uint8 = exposure.rescale_intensity(I_skimage, out_range=(0, 255)).astype(np.uint8)
    match channel:
        case 1:
            #图像拼接
            combined_image1 = np.hstack((fftImage_uint8, I_skimage_uint8))
            combined_image2 = np.hstack((reImage_uint8, timeHistogram_uint8))
            combined_image = np.vstack((combined_image1, combined_image2))
            combined_image=skimage_resize(combined_image, target_shape, order=3, anti_aliasing=True, preserve_range=True)
        case 3:
            # 拼接为 (C, H, W)
            combined_image = np.stack([fftImage_uint8, I_skimage_uint8, reImage_uint8], axis=0)
        case 4:
            combined_image = np.stack([fftImage_uint8, I_skimage_uint8, reImage_uint8,timeHistogram_uint8], axis=0)
        case _:
            raise ValueError(f"无效的 channel: {channel}")
    return combined_image

def in_outFilename(dataType):
    sharePath="/mnt/hgfs/winShare/"
    match dataType:
        case 1:
            filename="05-29-20222043_000_unp_2.bin" 
            interfType="CW"
        case 2:
            filename="05-29-20222018_000_unp_2.bin" 
            interfType="SCW(0.1023)"
        case 3:
            filename="05-29-20222027_000_unp_2.bin" 
            interfType="SCW(1.023)"
        case 4:
            filename="05-29-20222033_000_unp_2.bin" 
            interfType="AGWN(0.1023)"
        case 5:
            filename="05-29-20222038_000_unp_2.bin" 
            interfType="AGWN(1.023)"
        case 6:
            filename="04-29-20221545_unp_2.bin" 
            interfType="PI(1ms_0.3)"
        case 7:
            filename="04-29-20221607_unp_2.bin" 
            interfType="PI(1ms_0.7)"
        case 8:
            filename="04-29-20221557_unp_2.bin" 
            interfType="PI(10ms_0.3)"
        case 9:
            filename="04-29-20221613_unp_2.bin" 
            interfType="PI(10ms_0.7)"
        case _:
            raise ValueError(f"无效的 dataType: {dataType}")
    file_path=f"{sharePath}{filename}"
    return file_path,interfType
def main(skipnum,dataType):
    
    # --- 参数设定 ---
    #file_path = "/mnt/hgfs/winShare/05-29-20222027_000_unp_2.bin"  
    #interfType="SCW(1.023)"                #干扰类型
    file_path,interfType=in_outFilename(dataType)
    target_dtype = np.int8         # 目标数据类型 
    data_len     =10               #用多少ms数据做STFT
    #skipnum      =10              #跳过的数据长度(s),跳过的数据长度最好选择10，40……
    nImages      =250              #生成的stft图片数,1000对应10s，每一个数据ISR变化间隔30s
    target_shape=(224,224)         #裁切的大小
    numofChannel=3                 #生成图片的通道维数
    savePath=f"/home/naixa/LINUX/OSR/imageSet_{numofChannel}/"
    os.makedirs(savePath, exist_ok=True)
    # ----------------
    ISR=int(skipnum/30)*10-20
    savename=f"{interfType}_{ISR}"
    s=settings()
    samplesPerCode=round(s.samplingFreq/(s.codeFreqBasis/s.codeLength))
    if target_dtype==np.uint8:
        dataBlockLen=int(data_len/2*samplesPerCode)
    elif target_dtype==np.int8:
        dataBlockLen=int(data_len*samplesPerCode)
    
    skip_offset = int(skipnum * s.samplingFreq)
    try:
        with open(file_path, 'rb') as f:
            # 从文件对象 f 读取数据
            # dtype: 指定数据类型
            # count: 指定要读取的元素数量
                f.seek(skip_offset)
                # 处理所有数组并存储到列表
                all_arrays = []
                for i in range(nImages):
                    if stop_event.is_set():
                        return 
                # --- 读取二进制文件 ---
                    data_array = np.array([], dtype=target_dtype) # 初始化为空，以防文件打不开或读取失败
                    data_array = np.fromfile(f, dtype=target_dtype, count=dataBlockLen)
                    combined_image=imageProcess(data_array,s,target_shape,numofChannel)
                    # plt.imshow(combined_image,'gray')
                    # plt.savefig('fft.png',dpi=400)
                    # exit(0)
                    all_arrays.append(combined_image)
                    if i % 10 == 0:
                        print(f"process:{i}/{nImages}")
                    #savefliename=f"{savePath}image{i+1}.png"
                    #plt.imsave(savefliename, I_skimage, cmap='gray', format='png')
                # 转换为三维数组 (n_images, height, width)
                all_arrays = [arr.astype(np.uint8) for arr in all_arrays]
                stacked_arrays = np.stack(all_arrays, axis=0)
                # 保存为单个NPY文件
                output_file = f"{savePath}{savename}.npy"
                np.save(output_file, stacked_arrays)
    except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 未找到。")
            exit(0)
    except Exception as e:
            print(f"读取文件时发生错误: {e}")
            exit(0)
#创建多线程
def threadProcess(skipnumList,dataType):
    #线程数组
    threads=[]
    for skipnum in skipnumList:
        thread = threading.Thread(target=main, args=(skipnum,dataType))
        threads.append(thread)
        thread.start()
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    print("所有线程已完成")

    
if __name__ == '__main__':
    dataType=int(sys.argv[1])
    #skipnumList=[10]
    skipnumList=[10 ,40 ,70 ,100, 130 ,160 ,190 ,220]
    start_time = datetime.now()
    print(f"开始时间: {start_time}")
    threadProcess(skipnumList,dataType)
    end_time = datetime.now()
    print(f"结束时间: {end_time}")
    print(f"总用时: {end_time - start_time}")
    