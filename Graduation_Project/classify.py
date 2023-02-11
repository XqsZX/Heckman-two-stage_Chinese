import pandas as pd
import numpy as np


class Data:
    def __init__(self, classify_path, CSMAR_path, year):
        self.Classify = {}
        self.CSMAR = {}
        data = pd.read_csv(CSMAR_path, encoding='utf-8-sig')
        data = data.dropna()
        data = data[data['EndDate'].str.contains(year)]
        m = len(data)
        for i in range(m):
            self.CSMAR[data.iloc[i, 0]] = data.iloc[i, 4]

        classify = pd.read_excel(classify_path)
        classify = classify.iloc[0:3549, 0:26]
        n = len(classify)
        for i in range(n):
            if classify.iloc[i, int(year) - 1993] == np.nan:
                continue
            else:
                self.Classify[classify.iloc[i, 0]] = classify.iloc[i, int(year) - 1993]

        print(self.CSMAR)
        print(len(self.CSMAR))
        print(self.Classify)
        print(len(self.Classify))


def main():
    classify_path = r'C:\Users\15245\Desktop\data\基础数据\历年行业分类汇总-修改版.xlsx'
    CSMAR_path = r'C:\Users\15245\Desktop\data\基础数据\STK_LISTEDCOINFOANL.csv'
    fenlei = Data(classify_path, CSMAR_path, '2000')


if __name__ == '__main__':
    main()
