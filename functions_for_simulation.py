## functions

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def Xs_generator(rowNum, colNum):
    mus = np.zeros(colNum)
    sigma = np.eye(colNum)
    Xs = np.random.multivariate_normal(mus, sigma, rowNum)
    strs = ['x'+ str(i) for i in range(colNum)]
    df = pd.DataFrame(Xs, columns=strs)
    return df


def Ys_generator(Xs, Ws, tau):

    rowNum, colNum = Xs.shape

    beta1 = [np.random.uniform(-1, 1) for i in range(colNum)]
    beta_norm2 = np.linalg.norm(beta1)
    desired_norm = 10
    beta1 = beta1 / beta_norm2 * desired_norm

    beta0 = np.random.uniform(-1, 1)
    eps = np.random.normal(0, 0.25, rowNum)
    Ys = beta0 + np.dot(Xs, beta1) + tau * Ws + eps

    Xs['w'] = Ws
    Xs['y'] = Ys
    
    return(Xs)

def Ws_generator_equal_size(df):

    rowNum= df.shape[0]

    if rowNum % 2 != 0:
        raise ValueError("The input number rowNum must be an even number.")

    vector = np.ones(rowNum, dtype=int)
    indices = np.random.permutation(rowNum)
    vector[indices[:rowNum // 2]] = 0 # 0 represents controled group

    return vector

# 待重构
# def tau_hat_calculator(Ys_0, Ys_1, Ws):
#     return (Ys_1 @ Ws) / np.sum(Ws) - (Ys_0 @ (1 - Ws)) / np.sum(1 - Ws)

def tau_hat_calculator(df, Ws):
    Ys = df['y']
    return (Ys @ Ws) / np.sum(Ws) - (Ys @ (1 - Ws)) / np.sum(1 - Ws)


# 待重构
def diff_T_C_calculator(Xs, Ws):
    cov_X = np.cov(Xs)
    n = len(Ws)
    pw = np.sum(Ws) / n
    return Xs @ (Ws - np.ones(n) * pw) / (n * pw * (1 - pw))

# 待重构
def Mahalanobis_distance(Xs, Ws):
    cov_X = np.cov(Xs)
    n = len(Ws)
    pw = np.sum(Ws) / n
    diff_T_C = Xs @ (Ws - np.ones(n) * pw) / (n * pw * (1 - pw))
    
    try:
        M = n*pw*(1-pw) * diff_T_C.T @ np.linalg.inv(cov_X) @ diff_T_C
    except:
        M = n*pw*(1-pw) / cov_X * np.dot(diff_T_C, diff_T_C)

    return M

# 待重构
def Ws_generator_rerandomization(Xs, a, ifPrint = True):
    n = len(Xs)
    Ws = Ws_generator_equal_size(n)
    M = Mahalanobis_distance(Xs, Ws) # 如果使用了其他距离准则可以重写此语句
    while M > a:
        Ws = Ws_generator_equal_size(n)
        M = Mahalanobis_distance(Xs, Ws)
    if ifPrint:
        print("----- Accepted Rerandomization ----- ")
        print("Mahalanobis_distance = {0:.3f} \na = {1}".format(M, a))
        print("Accepted Ws: {0}".format(Ws))
    return Ws

# 待重构
def Ws_generator_AAtest(Xs, alpha=0.05, ifPrint = True):
    n = len(Xs)
    flag = True
    while flag == True:
        Ws = Ws_generator_equal_size(n)
        Ws_0 = np.zeros(n)
        Ys = Ys_generator(Xs, Ws_0)
        Ys_T = [Ys[i] for i in range(n) if Ws[i] == 1]
        Ys_C = [Ys[i] for i in range(n) if Ws[i] == 0]
        t_stat, p_value = stats.ttest_ind(Ys_T, Ys_C, equal_var=True)
        # 输出t统计量和p值
        if ifPrint:
            print(f"t-statistic: {t_stat}, p-value: {p_value}") 
        # 判断p值是否小于置信水平
        if p_value < alpha:
            print("拒绝原假设，即两个样本的均值不相等, Ws:{0}。".format(Ws))
        else:
            print("不能拒绝原假设，即没有足够的证据表明两个样本的均值不相等。")
            flag = False
            return Ws

# 待重构
def randomization_output_plot(Xs, Ws, figsize=(12, 5)):

    n = len(Xs)
    plt.figure(figsize=figsize)
    plt.xticks(range(n))
    # plt.yticks(range(-8, 9))
    plt.title('your title')
    plt.xlabel('index')
    plt.ylabel('Xs')

    # color = ['red' if x == 1 else 'black' for x in Ws]
    # print(color)
    index_treatment = [i for i in range(n) if Ws[i] == 1]
    Xs_treatment = Xs[index_treatment]
    index_control = [i for i in range(n) if Ws[i] == 0]
    Xs_control = Xs[index_control]


    plt.scatter(index_treatment, Xs_treatment,  color='red', label = 'treatment')
    plt.legend()

    plt.scatter(index_control, Xs_control,  color='black', label = 'controled')
    plt.legend()

    plt.axhline(y=0, color='grey', linestyle='--', linewidth = 0.7)
    plt.show()

# 待重构
def Xs_clarify(df, clarify_crition):

    for key_item, key_value in clarify_crition.items():
        df[key_item] = df[key_item].apply(lambda x: '>{0:.2f}'.format(key_value) if x > key_value else '<{0:.2f}'.format(key_value))
    
    return df

# 待重构
def Ws_generator_stratified(Xs, clarify_crition, n_groups=2):

    df = pd.DataFrame({'ID': range(1, len(Xs)+1), 'Xs': Xs})
    
    df = Xs_clarify(df, clarify_crition)
    
    stratify_cols = df.columns.tolist()
    stratify_cols.pop(0)
    # 创建一个空的DataFrame来保存分层随机化的结果
    stratified_df = pd.DataFrame(columns=df.columns.tolist() + ['Group'])
    
    # 对于每个层次组合，进行随机分组
    for _, group in df.groupby(stratify_cols):
        # 随机生成分组信息
        group['Group'] = np.random.choice(range(0, n_groups), len(group), replace=True)
        # print(group)
        
        # 将分组信息追加到结果DataFrame中
        stratified_df = pd.concat([stratified_df, group])
        res = stratified_df.sort_values(by='ID', ascending=True)

    return res['Group'].tolist()

        






