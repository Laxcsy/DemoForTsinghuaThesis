import pandas as pd
import numpy as np

def get_range(min_value,max_value, lis):
    surface_class_name = {
        1: 'Ⅰ',
        2: 'Ⅱ',
        3: 'Ⅲ',
        4: 'Ⅳ',
        5: 'Ⅴ'
    }

    i = 0
    while (lis[i] < min_value) & (i < len(lis) - 1):
        i += 1
    if i == len(lis)-1:
        range_list = [lis[-1]]
        cls = len(lis) + 1
    else:
        j = 0
        while (lis[j] < max_value) & (j < len(lis) - 1):
            j += 1

        if lis[-1] < max_value:
            cls = len(lis) + 1
        else:
            cls = j + 1

        if j == len(lis) - 1:
            range_list = lis[:] if i == 0 else lis[i-1:]
        else:
            range_list = lis[:j+1] if i == 0 else lis[i-1: j+1]

    range_indexes = [surface_class_name[i+1] for i,x in enumerate(lis) if x in range_list]

    return range_list, range_indexes, cls

class IndexStandard(object):
    def __init__(self, file):
        self.surface_water = pd.read_excel(file,sheet_name='surface_water')
        self.drinking_water = pd.read_excel(file,sheet_name='drinking_water')

    def get_class(self, index, value_list):
        value_list = np.asarray(value_list)
        min_value = np.min(value_list)*0.8
        max_value = np.max(value_list)*1.2

        if index in list(self.surface_water['index']):
            surface_standard = self.surface_water[self.surface_water['index']==index].values.flatten().tolist()[1:]
            surface_range, surface_indexes, surface_clss = get_range(min_value,max_value,surface_standard)

        else:
            surface_range = None
            surface_indexes = None
            surface_clss = None

        if index in list(self.drinking_water['index']):
            drinking_standard = list(self.drinking_water[self.drinking_water['index']==index]['limit'])[0]
            if (drinking_standard > min_value) | (drinking_standard < max_value):
                drinking_range = [drinking_standard]
            else:
                drinking_range = None
        else:
            drinking_range = None

        return surface_range, surface_indexes, drinking_range, surface_clss

if __name__ == '__main__':
    WaterStandard = IndexStandard('/Users/liaoxin/Documents/Tsinghua/data/water_standard/water_standard.xlsx')
    NH3_N = [
        0.405362027,
        0.502809591,
        0.569142962,
        0.449867754,
        0.451519402,
        0.606950328,
        0.395063465,
        0.454801097,
        0.590609122,
        0.403341714,
        0.420345684,
        0.584791974,
        0.54093963,
        0.546533042,
        0.691514261,
        0.583225819,
        0.51073521,
        0.565103167,
        0.509840264,
        0.533108855,
        0.525501815,
        0.490598929,
        0.495297395,
        0.555035027
    ]
    print(WaterStandard.get_class('NH3-N',NH3_N))

    min_value = 0.72
    max_value = 2.5
    lis = [0.15, 0.5, 1, 1.5, 2]
    print(get_range(min_value, max_value, lis))