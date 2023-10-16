import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import os
from scipy.interpolate import griddata

class EEmDataset(object):
    def __init__(self, data_path, exp_path = None):
        self.data_path = data_path
        self.files = glob(f'{self.data_path}*.TXT')
        self.samples = [x.split('.')[0] for x in list(map(os.path.basename,self.files))]
        self.blanks = glob(f'{self.data_path}*-w(FD3).TXT')
        self.times = list(dict.fromkeys([int(x.split('-')[0]) for x in self.samples]))
        self.times.sort()
        self.intergration = {}
        self.proportion = {}
        self.FL_indexes = {}
        for t in self.times:
            self.intergration[t] = {}
            self.proportion[t] = {}
            self.FL_indexes[t] = {}
        if exp_path:
            self.exp_path = exp_path
            check_path(self.exp_path)
        else:
            self.exp_path = None
        self.load_EEM()

    def load_EEM(self):
        eem_processor = EEmProcessor()
        for i, file in enumerate(self.files):
            name = file.split('(')[0]
            name = name.split('/')[-1]
            time = int(name.split('-')[0])
            if name.split('-')[1]!='w':
                pos = int(name.split('-')[1])
                EX,EM,FL= self.read_EEM(file)
                # get blank FL
                blank_sample = [x for x in self.blanks if str(time) in x][0]
                _, _, FL0 = self.read_EEM(blank_sample)
                FL = FL - FL0
                EX,EM,FL = eem_processor.scatter_remove((EX,EM,FL))
                FL = eem_processor.miss_value_interpolation(EX,EM,FL)
                FL = eem_processor.smooth_eem(EX,EM,FL)
                FI, BIX, HIX = eem_processor.calculate_indexes(EX.tolist(),EM.tolist(),FL)
                self.drawContourMap(EX,EM,FL,self.samples[i])
                self.intergration[time][pos] = self.intergrate(EX,EM,FL)
                self.proportion[time][pos] = [x/sum(self.intergration[time][pos]) for x in self.intergration[time][pos]]
                self.FL_indexes[time][pos] = [FI, BIX, HIX]
            else:
                continue

    def read_EEM(self,input_file):
        with open(input_file) as f:
            lines = f.readlines()
            index = [idx for idx, s in enumerate(lines) if 'Data points' in s][0]
            temp = [ x.replace('\n', "") for x in lines[index +1:][0].split('\t') ]
            arr = np.zeros((len(lines) - (index +1) ,len(temp)))
            r = 0
            for row in range(index +1 ,len(lines)):
                l = [ x.replace('\n', "") for x in lines[row:][0].split('\t') ]
                if r == 0:
                    arr[r ,1:] = list(map(float ,l[1:]))
                else:
                    arr[r ,:] = list(map(float ,l))
                r+=1

        f.close()

        EX = (arr[0])[1:]  # 第一行中除去第一个元素 EX作X
        EM = ((arr.T)[0])[1:]  # 第一列中除去第一个元素 EM 作 Y
        FL = ((arr[1:].T)[1:]).T  # 除去第一行和第一列 fl 作 数据
        return (EX,EM,FL)

    def intergrate(self,EX,EM,FL):
        Zone = {
            1: [220,250,280,330],
            2: [220,250,330,380],
            3: [220,250,380,500],
            4: [250,280,280,380],
            5: [250,400,380,500]
        }
        area = {}
        for key,val in Zone.items():
            area[key] = (val[1]-val[0])*(val[3]-val[2])

        area_mf = {k:sum(area.values())/v for k,v in area.items()}

        Intergration = []
        for Z in Zone.keys():
            Ex_indexes = find_indexes(EX, Zone[Z][0], Zone[Z][1])
            Em_indexes = find_indexes(EM, Zone[Z][2], Zone[Z][3])

            s = 0
            for i in Ex_indexes:
                for j in Em_indexes:
                    s += FL[j,i] * 1 * 5

            Intergration.append(s)

        return np.multiply(np.asarray(Intergration),np.asarray(list(area_mf.values())))

    def drawContourMap(self, X, Y, Z,fileName, Z_MIN=0, Z_MAX=10000, Z_STEP=100, LINE_STEP=1000):
        '''
            XYZ画图参数  fileName保存成文件的文件名  toPath 保存文件的路径
        '''
        #将原始数据变成网格数据形式
        X, Y = np.meshgrid(X, Y)
        N = np.arange(Z_MIN, Z_MAX, Z_STEP)
        CS = plt.contourf(X,Y,Z,N,cmap = mpl.cm.jet)
        plt.colorbar(CS)
        plt.contour(X,Y,Z,LINE_STEP, cmap = mpl.cm.jet)
        if self.exp_path:
            plt.savefig(self.exp_path + fileName + '.jpg')
        plt.close()
        plt.show()

class EEmProcessor(object):
    def __init__(self):
        pass

    def scatter_remove(self,sample):
        '''
        :param sample: a list of [EX,EM,FL]
        :return:
        '''

        EX = sample[0]  # 取ex
        EM = sample[1]  # 取em
        FL = sample[2]  # 取 fl

        # 第二步根据ex和em找到散射峰位置然后直接去散射
        FL = FL.T
        for i in range(len(FL)):  # 遍历ex
            for j in range(len(FL[i])):  # 遍历em
                ex = EX[i]
                em = EM[j]
                if (ex >= em - 20 and ex <= em + 20):  # and ex <= em + 20
                    FL[i, j] = None  # 不能使用0, 因为0是有意义的数据
                if (em >= 2 * ex - 20 and em <= 2 * ex + 20):  # and em <= 2 * ex + 20
                    FL[i, j] = None  # 不能使用0, 因为0是有意义的数据
                if (em < 1.55 * ex - 190 and em > 1.55 * ex - 240):  # and ex > 380 and em <= 2 * ex + 20     1.4x - 115
                    FL[i, j] = None  # 不能使用0, 因为0是有意义的数据
        FL = FL.T

        return EX,EM,FL

    def miss_value_interpolation(self, EX, EM, FL):
        """
        Use Delaunay triangulation method to interpolate in place, aim at removing nans in EEMs so
        that the parafac program can run normally.
            Reference DOI: 10.1016/j.marchem.2004.02.006
        """
        x, y = np.meshgrid(EX, EM)
        nan_ed = np.where(~np.isnan(FL.astype('float')))
        FL = griddata((EX[nan_ed[1]], EM[nan_ed[0]]), FL[nan_ed], (x, y), method='cubic')
        FL[FL < 0] = 0

        return FL

    def smooth_eem(self,EX, EM, FL, sigma=0.5, filter_size=None):
        """
        Use Gaussian kernel to smooth EEMs in place. Suggesting after interpolation.

        :param sigma: Optional, standard deviation of the Gaussian distribution, specified as a positive number.
            Default is 0.5.
        :param filter_size: optional, size of the Gaussian filter, specified as a positive,
            odd integer. The default filter size is 2*ceil(2*sigma)+1.
        """
        if filter_size is None:
            filter_size = 2 * int(np.ceil(2 * sigma)) + 1
        kernel = np.zeros((filter_size, filter_size))
        center = filter_size // 2
        for i in range(filter_size):
            for j in range(filter_size):
                x, y = center - i, center - j
                kernel[i, j] = -(x ** 2 + y ** 2) / (2 * sigma ** 2)
        kernel = np.exp(kernel)
        kernel /= np.sum(kernel)
        t = np.zeros((len(EM), len(EX)))

        xx = np.pad(FL, center, mode='reflect')
        for k in range(len(EM)):
            for q in range(len(EX)):
                t[k, q] = np.sum(xx[k:k + filter_size, q:q + filter_size] * kernel)
        FL = t

        return FL

    def calculate_indexes(self, EX, EM, FL):
        FI_EX = EX.index(370)
        FI_EM_1 = EM.index(450)
        FI_EM_2 = EM.index(500)

        BIX_EX = EX.index(310)
        BIX_EM_1 = EM.index(380)
        BIX_EM_2 = EM.index(430)

        HIX_EX = EX.index(255)
        HIX_EM_1 = find_indexes(EM, 435, 480)
        HIX_EM_2 = find_indexes(EM, 300, 345)

        Val1 = 0
        Val2 = 0
        for EM_1 in HIX_EM_1:
            Val1 += FL[EM_1, HIX_EX]
        for EM_2 in HIX_EM_2:
            Val2 += FL[EM_2, HIX_EX]

        FI = FL[FI_EM_1, FI_EX] / FL[FI_EM_2, FI_EX]
        BIX = FL[BIX_EM_1, BIX_EX] / FL[BIX_EM_2, BIX_EX]
        HIX = Val1/Val2

        return FI, BIX, HIX

    def martrix_to_DF(self, ex, em, fl):
        fl = np.insert(fl, 0, ex, 0)
        fl = fl.T
        em = np.insert(em, 0, 0)
        fl = np.insert(fl, 0, em, 0)
        fl = fl.T
        df = pd.DataFrame(fl)

        return df

def find_indexes(l, min, max):
    indexes = [
        index for index in range(len(l))
        if ((l[index] >= min) & (l[index]<=max))
    ]

    return indexes

def check_path(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Successfully created {path}!")

    return None