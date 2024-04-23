import numpy as np
import imageio
import os
import pickle

def load_cali_info(Cali,Mat, prnu_path,img_path,dark_path,exp_list):
    Cali.gain=Mat[0]
    Cali.rd_mean = Mat[1]
    Cali.rd_var = Mat[2]
    Cali.vsat = Mat[3]
    Cali.exptime = exp_list

    if prnu_path !=None:
        Cali.prnu_sign=True
    Cali.flat = np.array(imageio.imread(prnu_path))
    img_pths = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.tif')])
    Cali.img = [np.array(imageio.imread(f)) for f in img_pths]
    dark_pths = sorted([os.path.join(dark_path, f) for f in os.listdir(dark_path) if f.endswith('.tif')])
    Cali.biasframe = [np.array(imageio.imread(f)) for f in dark_pths]

class Cali_FIle:
    def __init__(self):
        self.gain = None
        self.rd_mean = None
        self.rd_var = None
        self.vsat = None
        self.vmax = None
        self.exptime =None

        self.prnu_sign = None

        self.flat = None
        self.img = None
        self.biasframe = None

if __name__ == '__main__':
    califile1 = Cali_FIle()

    Cali_mat = [[0.9212,0.9431,0.9171],[32.0021,31.7340,31.9705],[18.0424,9.2935,17.995],[1023,1023,1023]]
    prnu_path = "D:\ldrs\grandos\calib\\aj.tif"
    img_path = "D:\ldrs\grandos\sample_v"
    dark_path = "D:\ldrs\grandos\sample_b"
    exp_list = [0.6667,0.1667,0.04,0.01,0.0025,0.000625]
    load_cali_info(califile1,Cali_mat,prnu_path,img_path,dark_path,exp_list)
    with open('califile.pkl', 'wb') as file:
        pickle.dump(califile1, file)
