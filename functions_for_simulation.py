## functions

import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt


def is_positive_definite(matrix):
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Check if all eigenvalues are positive
    eigenvalues, _ = np.linalg.eig(matrix)
    if np.all(eigenvalues > 0):
        return True
    else:
        return False

def Xs_generator(rowNum, colNum):
    mus = np.zeros(colNum)
    sigma = np.eye(colNum)
    Xs = np.random.multivariate_normal(mus, sigma, rowNum)
    strs = ['x'+ str(i) for i in range(colNum)]
    df = pd.DataFrame(Xs, columns=strs)
    return df


def Ys_generator(Xs, Ws, tau=1, desired_norm = 10):

    df = Xs.copy()
    rowNum, colNum = df.shape

    beta1 = [np.random.uniform(-1, 1) for i in range(colNum)]
    beta_norm2 = np.linalg.norm(beta1)
    beta1 = beta1 / beta_norm2 * desired_norm

    beta0 = np.random.uniform(-1, 1)
    eps = np.random.normal(0, 0.25, rowNum)
    Ys = beta0 + np.dot(Xs, beta1) + tau * Ws + eps

    df['w'] = Ws
    df['y'] = Ys
    
    return(df)

def Ws_generator_equal_size(Xs):

    rowNum = Xs.shape[0]

    if rowNum % 2 != 0:
        raise ValueError("The input number rowNum must be an even number.")

    vector = np.ones(rowNum, dtype=int)
    indices = np.random.permutation(rowNum)
    vector[indices[:rowNum // 2]] = 0 # 0 represents controled group

    return vector

def tau_hat_calculator(df):
    Ys = df['y']
    Ws = df['w']
    return (Ys @ Ws) / np.sum(Ws) - (Ys @ (1 - Ws)) / np.sum(1 - Ws)



def diff_T_C_calculator(Xs, Ws):
    n = len(Ws)
    pw = np.sum(Ws) / n
    return Xs.T @ (Ws - np.ones(n) * pw) / (n * pw * (1 - pw))

def Mahalanobis_distance(Xs, Ws):
    
    cov_X = Xs.cov()
    # print("cov_X:{0}".format(is_positive_definite(cov_X)))
    # print("cov_X's inverse:{0}".format(is_positive_definite(sc.linalg.pinv(cov_X))))
    n = len(Ws)
    pw = np.sum(Ws) / n
    diff_T_C = Xs.T @ (Ws - np.ones(n) * pw) / (n * pw * (1 - pw)) 
    # print("pw: {0}\ndiff_T_C: {1}".format(pw,diff_T_C))
    
    # try:
    M = n*pw*(1-pw) * diff_T_C.T @ sc.linalg.pinv(cov_X) @ diff_T_C
    # except:
    #     M = n*pw*(1-pw) / cov_X * np.dot(diff_T_C, diff_T_C)

    return M

def Ws_generator_rerandomization(Xs, a, ifPrint = True):
    Ws = Ws_generator_equal_size(Xs)
    M = Mahalanobis_distance(Xs, Ws) # 如果使用了其他距离准则可以重写此语句
    while M > a:
        if ifPrint:
            print("----- Unaccepted Rerandomization ----- ")
            print("Ws: {0}".format(Ws))
            print("Mahalanobis_distance = {0:.3f} \na = {1}".format(M, a))
        Ws = Ws_generator_equal_size(Xs)
        M = Mahalanobis_distance(Xs, Ws)
    if ifPrint:
        print("----- Accepted Rerandomization ----- ")
        print("Mahalanobis_distance = {0:.3f} \n threshold: a = {1}".format(M, a))
        print("Accepted Ws: {0}".format(Ws))
    return Ws

def Ws_generator_AAtest(Xs, alpha=0.1, ifPrint = True):
    
    rowNum = Xs.shape[0]
    flag = True
    while flag == True:
        Ws = Ws_generator_equal_size(Xs)
        Ws_0 = np.zeros(rowNum)
        Ys_0 = Ys_generator(Xs, Ws_0)['y']
        Ys_T = [Ys_0[i] for i in range(rowNum) if Ws[i] == 1]
        Ys_C = [Ys_0[i] for i in range(rowNum) if Ws[i] == 0]
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

def randomization_output_plot(df, figsize=(18, 5)):

    Ys = df['y']
    Ws = df['w']
    rowNum = df.shape[0]
    plt.figure(figsize=figsize)
    # plt.xticks(range(rowNum))
    # plt.yticks(range(-8, 9))
    plt.title('Ys: treatment vs control')
    plt.xlabel('index')
    plt.ylabel('Ys')

    # color = ['red' if x == 1 else 'black' for x in Ws]
    # print(color)
    index_treatment = [i for i in range(rowNum) if Ws[i] == 1]
    Ys_treatment = Ys[index_treatment]
    index_control = [i for i in range(rowNum) if Ws[i] == 0]
    Ys_control = Ys[index_control]


    plt.scatter(index_treatment, Ys_treatment,  color='red', label = 'treatment')
    plt.legend()

    plt.scatter(index_control, Ys_control,  color='black', label = 'controled')
    plt.legend()

    plt.axhline(y=0, color='grey', linestyle='--', linewidth = 0.7)
    plt.show()


def Xs_clarify(df, clarify_crition):

    df_new = df.copy()
    for key_item, key_value in clarify_crition.items():
        df_new[key_item] = df_new[key_item].apply(lambda x: '>{0:.2f}'.format(key_value) if x > key_value else '<{0:.2f}'.format(key_value))
    
    return df_new

def Ws_generator_stratified(Xs, clarify_crition, n_groups=2):

    # df = pd.DataFrame({'ID': range(1, len(Xs)+1), 'Xs': Xs})
    df = Xs.copy()
    df['ID'] = range(1, len(Xs)+1)
    df = Xs_clarify(df, clarify_crition)
    
    # stratify_cols = df.columns.tolist()
    # stratify_cols.pop(0)
    stratify_cols = list(clarify_crition.keys())
    # 创建一个空的DataFrame来保存分层随机化的结果
    stratified_df = pd.DataFrame(columns=df.columns.tolist() + ['Group'])
    
    # 对于每个层次组合，进行随机分组
    for key, group in df.groupby(stratify_cols):
        # print(key)
        # 随机生成分组信息
        group_new = group.copy()
        group_new['Group'] = np.random.choice(range(0, n_groups), len(group), replace=True)
        # res_group = np.random.choice(range(0, n_groups), len(group), replace=True)
        # res_group = pd.DataFrame({'Group': res_group})
        # group = pd.concat([group, res_group], axis=1)
        # print(group)
        
        # 将分组信息追加到结果DataFrame中
        stratified_df = pd.concat([stratified_df, group_new])
        res = stratified_df.sort_values(by='ID', ascending=True)

    return res['Group'].tolist()

        






