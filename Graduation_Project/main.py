import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.discrete.discrete_model import Probit


class KMV:
    def __init__(self, file_kmv, file_term, file_R, file_rate, year):
        self.K = {}
        self.V = {}
        self.u = {}
        self.sigma = {}
        self.Term = {}
        self.LR = {}
        self.KMV = {}
        self.Rate0 = 0
        self.Rate1 = 0
        self.Rate_L = 0

        kmv = pd.read_csv(file_kmv, encoding='utf-8-sig')
        kmv = kmv[kmv['MarketValueOfDebt'] != 0]
        kmv = kmv.dropna()
        kmv = kmv[kmv['Enddate'].str.contains(year)]
        m = len(kmv)
        for i in range(m):
            self.K[kmv.iloc[i, 0]] = kmv.iloc[i, 5]
            self.V[kmv.iloc[i, 0]] = kmv.iloc[i, 6]
            self.u[kmv.iloc[i, 0]] = kmv.iloc[i, 7]
            self.sigma[kmv.iloc[i, 0]] = kmv.iloc[i, 8]

        term = pd.read_csv(file_term, encoding='utf-8-sig')
        term = term.dropna()
        term = term[term['Accper'].str.contains(year)]
        n = len(term)
        # 如果长期贷款比率大于0.5，则认为该公司贷款主要由长期贷款组成，“Term”值取1，否则取0
        for i in range(n):
            if term.iloc[i, 3] > 0.5:
                self.Term[term.iloc[i, 0]] = 1
            else:
                self.Term[term.iloc[i, 0]] = 0
            self.LR[term.iloc[i, 0]] = term.iloc[i, 2]

        r = pd.read_csv(file_R, encoding='utf-8-sig')
        r = r[r['Clsdt'].str.contains(year)]
        self.Rate = r['Nrrdata'].mean()

        Rate0 = []
        Rate1 = []
        # 提取“Term”的值为0的公司代码
        for key in self.Term:
            if self.Term[key] == 0:
                Rate0.append(key)
            else:
                Rate1.append(key)

        rate = pd.read_csv(file_rate, encoding='utf-8-sig')
        rate = rate.drop_duplicates('Stkcd', keep='last')
        rate = rate[rate['A002114000'] > 0]
        rate = rate[rate['A002100000'] > 0].dropna()
        count0 = 0
        count1 = 0
        sum0 = 0
        sum1 = 0
        n = len(rate)
        for i in range(n):
            if rate.iloc[i, 0] in Rate0:
                sum0 = sum0 + rate.iloc[i, 3] / rate.iloc[i, 4]
                count0 = count0 + 1
            elif rate.iloc[i, 0] in Rate1:
                sum1 = sum1 + rate.iloc[i, 3] / rate.iloc[i, 4]
                count1 = count1 + 1
        self.Rate0 = sum0 / count0
        self.Rate1 = sum1 / count1
        print(self.Rate0)
        print(self.Rate1)
        print(self.Rate)

    def calc_kmv(self):
        for i in self.sigma.keys():
            if self.Term[i]:
                self.KMV[i] = np.log(self.V[i] / self.K[i]) + (self.u[i] - 0.5 * self.sigma[i] * self.sigma[i]) * 3 / \
                              self.sigma[i] * math.sqrt(3)
            else:
                self.KMV[i] = np.log(self.V[i]/self.K[i])-(self.u[i]-0.5*self.sigma[i]*self.sigma[i])/self.sigma[i]

    def get_kmv(self, i):
        return self.KMV[i]

    def get_term(self, j):
        return self.Term[j]

    def get_k(self, m):
        return self.K[m]

    def get_lr(self, n):
        return self.LR[n]

    def get_Rate0(self):
        return self.Rate0

    def get_Rate1(self):
        return self.Rate1


class Company:
    def __init__(self, file_SA, file_LASP, file_investment, year):
        self.size = {}
        self.SA = {}
        self.LASP = {}
        self.investment = {}
        self.investment2 = {}

        investment = pd.read_csv(file_investment, encoding='utf-8-sig')
        investment = investment.dropna()
        investment2 = investment[investment['EndDate'].str.contains(str(int(year) - 2))]
        investment = investment.drop_duplicates('Symbol', keep='last')
        investment2 = investment2.drop_duplicates('Symbol', keep='last')
        p = len(investment)
        q = len(investment2)
        for i in range(p):
            self.investment[investment.iloc[i, 0]] = investment.iloc[i, 2]
        for j in range(q):
            self.investment2[investment2.iloc[j, 0]] = investment2.iloc[j, 2]
        # print(self.investment)
        # print(self.investment2)

        sa = pd.read_csv(file_SA, encoding='utf-8-sig')
        sa = sa[sa['Enddate'].str.contains(year)]
        m = len(sa)
        for i in range(m):
            self.size[sa.iloc[i, 0]] = sa.iloc[i, 5]
            self.SA[sa.iloc[i, 0]] = sa.iloc[i, 6]

        lasp = pd.read_csv(file_LASP, encoding='utf-8-sig')
        lasp = lasp[lasp['Reptdt'].str.contains('-12-31')]
        lasp = lasp[lasp['Reptdt'].str.contains(year)]
        n = len(lasp)
        for i in range(n):
            self.LASP[lasp.iloc[i, 0]] = lasp.iloc[i, 2]

    def get_size(self, i):
        return self.size[i]

    def get_sa(self, j):
        return self.SA[j]

    def get_lasp(self, k):
        return self.LASP[k]

    def get_investment(self, p):
        return self.investment[p]

    def get_investment2(self, q):
        return self.investment2[q]


class Rating:
    def __init__(self, file_rating, file_T1, file_T4, file_T5, file_T6, file_T8, year):
        self.Rating = {}
        self.cash_ratio = {}
        self.ROA = {}
        self.RTR = {}
        self.N_C = {}
        self.EBIT = {}
        self.cash_flow = {}
        self.guaranteed_ratio = {}

        rate = pd.read_csv(file_rating, encoding='utf-8-sig')
        rate[rate == 0] = np.nan
        rate = rate.dropna()
        # 以每年最后一次公告所给出的信用评级作为当年的评级
        rate = rate[rate['RatingDate'].str.contains(year)]
        rate = rate.drop_duplicates('Symbol', keep='last')
        m = len(rate)
        for i in range(m):
            self.Rating[rate.iloc[i, 0]] = rate.iloc[i, 3]

        T1 = pd.read_csv(file_T1, encoding='utf-8-sig')
        T4 = pd.read_csv(file_T4, encoding='utf-8-sig')
        T5 = pd.read_csv(file_T5, encoding='utf-8-sig')
        T6 = pd.read_csv(file_T6, encoding='utf-8-sig')
        T8 = pd.read_csv(file_T8, encoding='utf-8-sig')

        T1 = T1[T1['Typrep'] == 'A']
        T1 = T1[T1['Accper'].str.contains(year)]
        T1 = T1.drop_duplicates('Stkcd', keep='last')
        n = len(T1)
        for i in range(n):
            self.cash_ratio[T1.iloc[i, 0]] = T1.iloc[i, 3]
            self.N_C[T1.iloc[i, 0]] = T1.iloc[i, 4]

        T4 = T4[T4['Typrep'] == 'A']
        T4 = T4[T4['Accper'].str.contains(year)]
        T4 = T4.drop_duplicates('Stkcd', keep='last')
        p = len(T4)
        for i in range(p):
            self.RTR[T4.iloc[i, 0]] = T4.iloc[i, 3]

        T5 = T5[T5['Typrep'] == 'A']
        T5 = T5[T5['Accper'].str.contains(year)]
        T5 = T5.drop_duplicates('Stkcd', keep='last')
        q = len(T5)
        for i in range(q):
            self.ROA[T5.iloc[i, 0]] = T5.iloc[i, 3]
            self.EBIT[T5.iloc[i, 0]] = T5.iloc[i, 4]

        T6 = T6[T6['Typrep'] == 'A']
        T6 = T6[T6['Accper'].str.contains(year)]
        T6 = T6.drop_duplicates('Stkcd', keep='last')
        w = len(T6)
        for i in range(w):
            self.cash_flow[T6.iloc[i, 0]] = T6.iloc[i, 3]

        T8 = T8[T8['Typrep'] == 'A']
        T8 = T8[T8['Accper'].str.contains(year)]
        T8 = T8.drop_duplicates('Stkcd', keep='last')
        k = len(T8)
        for i in range(k):
            self.guaranteed_ratio[T8.iloc[i, 0]] = T8.iloc[i, 3]

    def get_rate(self, i):
        return self.Rating[i]

    def get_cash_ratio(self, i):
        return self.cash_ratio[i]

    def get_ROA(self, i):
        return self.ROA[i]

    def get_RTR(self, i):
        return self.RTR[i]

    def get_N_C(self, i):
        return self.N_C[i]

    def get_EBIT(self, i):
        return self.EBIT[i]

    def get_cash_flow(self, i):
        return self.cash_flow[i]

    def get_guaranteed_ratio(self, i):
        return self.guaranteed_ratio[i]


class R:
    def __init__(self, file_R, year):
        self.R = 0
        r = pd.read_csv(file_R, encoding='utf-8-sig')
        r = r[r['Clsdt'].str.contains(year)]
        self.R = r['Nrrdata'].mean()

    def get_R(self):
        return self.R


class Data:
    def __init__(self, file_kmv, file_term, file_SA, file_LASP, file_investment, file_rating, file_R, file_rate,
                 file_T1, file_T4, file_T5, file_T6, file_T8, file_data):
        self.data = pd.DataFrame(columns=['Symbol', 'year', 'KMV', 'Term', 'Debt', 'Debt2', 'LR', 'size', 'SA', 'LASP',
                                          'R', 'cash_flow', 'investment_growth', 'guaranteed_ratio', 'N_C', 'cash_ratio'
                                          , 'ROA', 'RTR', 'EBIT', 'Rating'])

        year = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                '2019']
        count = 0
        for i in year:
            kmv = KMV(file_kmv=file_kmv, file_term=file_term, file_R=file_R, file_rate=file_rate, year=i)
            kmv.calc_kmv()
            company = Company(file_SA=file_SA, file_LASP=file_LASP, file_investment=file_investment, year=i)
            rating = Rating(file_rating, file_T1, file_T4, file_T5, file_T6, file_T8, year=i)
            r = R(file_R=file_R, year=i)

            print("processing the data in year: ", i)
            for j in kmv.KMV.keys() & company.investment2.keys():
                line_na = pd.DataFrame(columns=self.data.columns, index=[count])
                self.data = pd.concat([self.data, line_na])
                self.data.iloc[count, 0] = j
                self.data.iloc[count, 1] = i
                self.data.iloc[count, 2] = abs(kmv.get_kmv(j))
                self.data.iloc[count, 3] = kmv.get_term(j)
                self.data.iloc[count, 4] = kmv.get_k(j) / (company.get_size(j) * 10e4)
                self.data.iloc[count, 5] = np.log(kmv.get_k(j) ** 2 / (company.get_size(j) * 10e4))
                self.data.iloc[count, 6] = kmv.get_lr(j)
                self.data.iloc[count, 7] = math.log(company.get_size(j))
                self.data.iloc[count, 8] = abs(company.get_sa(j))
                self.data.iloc[count, 9] = company.get_lasp(j)
                if kmv.get_term(j) == 0:
                    self.data.iloc[count, 10] = kmv.get_Rate0() * 100
                else:
                    self.data.iloc[count, 10] = kmv.get_Rate1() * 100
                # self.data.iloc[count, 10] = r.get_R()
                self.data.iloc[count, 11] = abs(rating.get_cash_flow(j) / (company.get_size(j) * 10e4))
                self.data.iloc[count, 12] = (company.get_investment(j) - company.get_investment2(j)) \
                                            / company.get_investment2(j)
                self.data.iloc[count, 13] = rating.get_guaranteed_ratio(j)
                self.data.iloc[count, 14] = rating.get_N_C(j)
                self.data.iloc[count, 15] = rating.get_cash_ratio(j)
                self.data.iloc[count, 16] = rating.get_ROA(j)
                self.data.iloc[count, 17] = rating.get_RTR(j)
                self.data.iloc[count, 18] = rating.get_EBIT(j) / (company.get_size(j) * 10e4)

                # print(count)
                count = count + 1
            self.data.index = self.data['Symbol']

            for k in rating.Rating.keys():
                if k in self.data['Symbol']:
                    if rating.get_rate(k) in ['AAA']:
                        self.data.loc[k, ['Rating']] = 9
                    elif rating.get_rate(k) in ['AAA-']:
                        self.data.loc[k, ['Rating']] = 8
                    elif rating.get_rate(k) in ['AA+']:
                        self.data.loc[k, ['Rating']] = 7
                    elif rating.get_rate(k) in ['AA']:
                        self.data.loc[k, ['Rating']] = 6
                    elif rating.get_rate(k) in ['AA-']:
                        self.data.loc[k, ['Rating']] = 5
                    elif rating.get_rate(k) in ['A+']:
                        self.data.loc[k, ['Rating']] = 4
                    elif rating.get_rate(k) in ['A']:
                        self.data.loc[k, ['Rating']] = 3
                    elif rating.get_rate(k) in ['A-']:
                        self.data.loc[k, ['Rating']] = 2
                    else:
                        self.data.loc[k, ['Rating']] = 1

        self.data = self.data[self.data['size'] <= 9]
        # self.data = self.data.dropna()
        self.data = self.data[self.data['KMV'].notna()]
        self.data = self.data[self.data['Debt'].notna()]
        self.data = self.data[self.data['Debt2'].notna()]
        self.data = self.data[self.data['LR'].notna()]
        self.data = self.data[self.data['SA'].notna()]
        self.data = self.data[self.data['LASP'].notna()]
        self.data = self.data[self.data['cash_flow'].notna()]
        self.data = self.data[self.data['investment_growth'].notna()]
        self.data = self.data[self.data['guaranteed_ratio'].notna()]
        self.data = self.data[self.data['N_C'].notna()]
        self.data = self.data[self.data['cash_ratio'].notna()]
        self.data = self.data[self.data['ROA'].notna()]
        self.data = self.data[self.data['RTR'].notna()]
        self.data = self.data[self.data['EBIT'].notna()]
        self.data.to_csv(file_data, encoding='utf-8-sig', index=False)

    def random_forest(self, file_data):
        # 将数据复制出来操作，避免影响原数据
        df = pd.read_csv(file_data, encoding='utf-8-sig')
        rate = df.iloc[:, 19]
        df = df.iloc[:, 13:18]

        # 确定训练集（数据完整），测试集（有缺失值）

        x_train = df[rate.notnull()]
        y_train = rate[rate.notnull()]
        x_fill = df[rate.isnull()]

        rfc = RandomForestRegressor()
        rfc.fit(x_train, y_train)

        # 用模型来预测填充值
        y_fill = rfc.predict(x_fill)
        self.data.loc[self.data.iloc[:, 19].isnull(), 'Rating'] = y_fill

        self.data.to_csv(file_data, encoding='utf-8-sig', index=False)


class Heckman_two_stage:
    def __init__(self, file_data):
        self.inverse_mills = None
        self.df = pd.read_csv(file_data, encoding='utf-8-sig')
        self.Term = self.df['Term']

    def first_stage_probit(self):
        step1model = sm.Probit(self.Term, sm.add_constant(self.df.iloc[:, 4:15]))
        step1res = step1model.fit(disp=False)
        print(step1res.summary())
        # View inputs as arrays with at least two dimensions
        step1_fitted = np.atleast_2d(step1res.fittedvalues).T
        # Compute the variance/covariance matrix
        # step1_varcov = step1res.cov_params()

        self.inverse_mills = norm.pdf(step1_fitted) / norm.cdf(step1_fitted)
        self.df['IMR'] = self.inverse_mills
        self.df = self.df.dropna()
        self.df.to_csv(r'C:\Users\15245\Desktop\data\test.csv', encoding='utf-8-sig')

    def second_stage_ols(self):
        X = sm.add_constant(self.df.iloc[:, 3:22].drop(columns=['N_C', 'cash_ratio', 'ROA', 'RTR', 'EBIT', 'Term', 'R']))
        print(X)
        model = sm.OLS(self.df['KMV'], X)
        results = model.fit()
        print(results.summary())


def main():
    file_kmv = r'C:\Users\15245\Desktop\data\BDT_FinDistMertonDD.csv'
    file_term = r'C:\Users\15245\Desktop\data\CSR_Finidx.csv'
    file_SA = r'C:\Users\15245\Desktop\data\BDT_FinConstSA.csv'
    file_LASP = r'C:\Users\15245\Desktop\data\HLD_CR.csv'
    file_rating = r'C:\Users\15245\Desktop\data\DEBT_BOND_RATING.csv'
    file_R = r'C:\Users\15245\Desktop\data\TRD_Nrrate.csv'
    file_T1 = r'C:\Users\15245\Desktop\data\FI_T1.csv'
    file_T4 = r'C:\Users\15245\Desktop\data\FI_T4.csv'
    file_T5 = r'C:\Users\15245\Desktop\data\FI_T5.csv'
    file_T6 = r'C:\Users\15245\Desktop\data\FI_T6.csv'
    file_T8 = r'C:\Users\15245\Desktop\data\FI_T8.csv'
    file_investment = r'C:\Users\15245\Desktop\data\EVA_TotalCost.csv'
    file_rate = r'C:\Users\15245\Desktop\data\FS_Combas.csv'
    file_data = r'C:\Users\15245\Desktop\data\DATA_r.csv'
    file_test = r'C:\Users\15245\Desktop\data\test.csv'
    # data = Data(file_kmv, file_term, file_SA, file_LASP, file_investment, file_rating, file_R, file_rate, file_T1,
    #             file_T4, file_T5, file_T6, file_T8, file_data)
    # data.random_forest(file_data)
    Heckman = Heckman_two_stage(file_data)
    Heckman.first_stage_probit()
    Heckman.second_stage_ols()


if __name__ == '__main__':
    main()
