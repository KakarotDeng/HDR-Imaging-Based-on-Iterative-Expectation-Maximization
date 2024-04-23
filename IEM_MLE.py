import numpy as np
import pickle
from Calibration_file import Cali_FIle
from scipy.ndimage import convolve, gaussian_filter
import exifread
import OpenEXR
import rawpy
import Imath
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def savehdrandshow(hdr):
    R = (hdr[..., 0]).astype(np.float32).tobytes()
    G = (hdr[..., 1]).astype(np.float32).tobytes()
    B = (hdr[..., 2]).astype(np.float32).tobytes()

    # 设置图像大小和每个通道的数据
    header = OpenEXR.Header(hdr.shape[1], hdr.shape[0])
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, float_chan) for c in "RGB"])
    out = OpenEXR.OutputFile("merged_image.exr", header)
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()
    file = OpenEXR.InputFile('merged_image.exr')
    header = file.header()

    # 获取图像的宽度和高度
    width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
    height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1

    # 读取RGB通道的数据
    R = np.frombuffer(file.channel('R'), dtype=np.float32)
    G = np.frombuffer(file.channel('G'), dtype=np.float32)
    B = np.frombuffer(file.channel('B'), dtype=np.float32)

    # 重塑数据为图像形状
    hdr_data = np.stack((R, G, B), axis=-1).reshape((height, width, 3))
    # hdr_data=hdr_data/np.max(hdr_data)
    # 对HDR图像进行色调映射
    # exposure_map = exposure.adjust_gamma(hdr_data, gamma=0.2)
    # #exposure_map[exposure_map > 1] = 1
    # # 将LDR图像数据保存为文件
    output_ldr_file = "output_ldr_image.jpg"
    # # 将LDR图像数据保存为文件，这里使用了skimage库中的exposure.adjust_log()函数进行色调映射
    # # 你也可以使用其他库或方法进行色调映射，例如OpenCV、PIL等
    # exposure.adjust_log(hdr_data)
    Tmed = cv2.createTonemapReinhard(0.8)
    ldr_image = Tmed.process(hdr_data)

    ldr_image = Image.fromarray((ldr_image * 255).astype(np.uint8))
    ldr_image.save(output_ldr_file)
    plt.imshow(ldr_image)
    plt.show()


def IE_mle():
    with open('califile.pkl', 'rb') as file:
        califile = pickle.load(file)
    stack_n = len(califile.exptime)

    gain = np.average(califile.gain)
    rd_mean = np.average(califile.rd_mean)
    rd_var = 0.3333*(califile.rd_var[0]+2*califile.rd_var[1]+califile.rd_var[2])
    Vmg = califile.img
    Bmg = califile.biasframe
    prnu_a = np.ones((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))

    if califile.prnu_sign != None:
        flat = califile.flat
        flat2 = gaussian_filter(flat, 2)
        prnu_a = flat / flat2

    Xivar = [0 for _ in range(6)]  # 辐照度
    Divar = [0 for _ in range(6)]  # 暗电流照度

    # 初始化辐照度权重
    for i in range(stack_n):
        Xi = gain * Vmg[i] - rd_mean
        Xi[Xi < 0] = 0
        Xivar[i] = (Xi + rd_var) / ((gain ** 2) * (califile.exptime[i] ** 2))
        Di = gain * Bmg[i] - rd_mean
        Di[Di < 0] = 0
        Divar[i] = (Di + rd_var) / ((gain ** 2) * (califile.exptime[i] ** 2))


    Xmeansnr = float('-inf')
    epoch = 0


    while True:
        Xmean = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))
        Dmean = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))
        Xvar = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))
        Dvar = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))
        ns = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))
        Xwsum = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))
        Dwsum = np.zeros((Vmg[0].shape[0], Vmg[0].shape[1], Vmg[0].shape[2]))

        for i in range(stack_n):

            Xmeani = (np.single(Vmg[i]) - np.single(Bmg[i])) / (gain * prnu_a * califile.exptime[i])
            Dmeani = (np.single(Bmg[i]) - rd_mean) / (gain * prnu_a * califile.exptime[i])

            Xwi = 1/Xivar[i]
            Dwi = 1 / Divar[i]

            Xmean += Xmeani*Xwi
            Dmean += Dmeani*Dwi
            Xvar += Xwi
            Dvar += Dwi
            Xwsum += Xwi
            Dwsum += Dwi
            ns += ns + Vmg[i]
        Xmean = Xmean/Xwsum
        Dmean = Dmean/Dwsum

        Xsnrhat = 20 * np.log10(Xmean / np.sqrt(Xvar))
        Xmeansnr0 = Xmeansnr
        Xmeansnr = np.min(Xsnrhat)
        print(Xmeansnr)

        epoch += 1
        if epoch > 5:
            break

        for i in range(stack_n):
            I = (prnu_a * Xmean + Dmean).copy()
            I[I < 0] = 0
            I2 = Dmean.copy()
            I2[I2 < 0] = 0
            Vvar2 = (gain ** 2) * califile.exptime[i] * I + rd_var
            Bvar2 = (gain ** 2) * califile.exptime[i] * I2 + rd_var
            Xvar2 = (Vvar2 + Bvar2) / ((califile.exptime[i] ** 2) * (gain ** 2) * (prnu_a ** 2))
            Dvar2 = (Bvar2 + rd_var) / ((califile.exptime[i] ** 2) * (gain ** 2))
            Xivar[i] = Xvar2
            Divar[i] = Dvar2
    return Xmean

if __name__ == '__main__':
    hdr = IE_mle()
    savehdrandshow(hdr)














