## functions

import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import chi2

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

def Xs_generator(rowNum, colNum, sigma_value=1):
    mus = np.zeros(colNum)
    sigma = sigma_value*np.eye(colNum)
    Xs = np.random.multivariate_normal(mus, sigma, rowNum)
    strs = ['x'+ str(i) for i in range(colNum)]
    df = pd.DataFrame(Xs, columns=strs)
    return df


def Ys_generator(Xs, Ws, tau=3, desired_norm = 10):

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

def Ys_generator_quadratic(Xs, Ws, tau=1, desired_norm = 10):

    df = Xs.copy()
    rowNum, colNum = df.shape

    beta1 = [np.random.uniform(-1, 1) for i in range(colNum)]
    beta_norm2 = np.linalg.norm(beta1)
    beta1 = beta1 / beta_norm2 * desired_norm

    beta0 = np.random.uniform(-1, 1)
    eps = np.random.normal(0, 0.25, rowNum)

    A = [np.random.uniform(-1, 1) for i in range(colNum)]
    A_norm2 = np.linalg.norm(A)
    A = A / A_norm2 * desired_norm
    A = np.diag(A)
    print(A.shape)

    Ys = beta0 + np.dot(Xs, beta1) + np.einsum('bi,ij,bj->b', Xs, A, Xs) + tau * Ws + eps

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
    M = n*pw*(1-pw) * diff_T_C.T @ np.linalg.pinv(cov_X) @ diff_T_C
    # M = n*pw*(1-pw) * diff_T_C.T @  diff_T_C
    # except:
    # M = n*pw*(1-pw) / cov_X * np.dot(diff_T_C, diff_T_C)

    return M

def Mahalanobis_distance_regularization(Xs, Ws, lambda_value=1):


    _, colNum = Xs.shape
    cov_X = Xs.cov()
    n = len(Ws)
    pw = np.sum(Ws) / n
    diff_T_C = Xs.T @ (Ws - np.ones(n) * pw) / (n * pw * (1 - pw)) 
    # print("pw: {0}\ndiff_T_C: {1}".format(pw,diff_T_C))
    
    # try:
    M = n*pw*(1-pw) * diff_T_C.T @ np.linalg.pinv(cov_X+np.eye(colNum)*lambda_value) @ diff_T_C

    return M


def l2_distance(Xs, Ws):


    _, colNum = Xs.shape
    cov_X = Xs.cov()
    n = len(Ws)
    pw = np.sum(Ws) / n
    diff_T_C = Xs.T @ (Ws - np.ones(n) * pw) / (n * pw * (1 - pw)) 
    # print("pw: {0}\ndiff_T_C: {1}".format(pw,diff_T_C))
    
    # try:
    M = diff_T_C.T @ diff_T_C

    return M

def choice_of_threshold(colNum, Pa):
    return chi2.ppf(Pa, colNum)

def Ws_generator_rerandomization(Xs, a, ifPrint = False):
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

def Ws_generator_AAtest(Xs, alpha=0.2, ifPrint = False):
    
    rowNum = Xs.shape[0]
    flag = True
    step = 0
    while flag == True:
        step += 1
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
            if ifPrint:
                print("拒绝原假设，即两个样本的均值不相等, Ws:{0}。".format(Ws))
        else:
            if ifPrint:
                print("不能拒绝原假设，即没有足够的证据表明两个样本的均值不相等。")
                print("----- AAtest Rerandomization ----- ")
                print('By {0} steps, we get an accepted Ws'.format(step))
            flag = False
            return Ws

def randomization_output_plot(df, linewidth = 0.2, figsize=(25, 5)):

    Ys = df['y']
    Ws = df['w']
    rowNum = df.shape[0]
    plt.figure(figsize=figsize)
    # plt.xticks(range(rowNum))
    # plt.yticks(range(-8, 9))
    plt.title(r'Randomization Output of $\mathbf{y}$: Treatment vs Control')
    plt.xlabel('Index')
    plt.ylabel('y')

    # color = ['red' if x == 1 else 'black' for x in Ws]
    # print(color)
    index_treatment = [i for i in range(rowNum) if Ws[i] == 1]
    Ys_treatment = Ys[index_treatment]
    index_control = [i for i in range(rowNum) if Ws[i] == 0]
    Ys_control = Ys[index_control]


    plt.scatter(index_treatment, Ys_treatment, color='#ffa500', label = 'treatment', edgecolors='black', linewidths=linewidth)
    plt.legend()

    plt.scatter(index_control, Ys_control,  color='#add8e6', label = 'controled', edgecolors='black', linewidths=linewidth)
    plt.legend()

    plt.axhline(y=0, color='grey', linestyle='--', linewidth = 0.7)
    plt.show()

def tau_hat_boxplot(method, iterNum, *args):

    Xs = args[0]

    methodset = {
        'AAtest': Ws_generator_AAtest,
        'block randomization': Ws_generator_block,
        'stratified randomization': Ws_generator_stratified,
        'rerandomization': Ws_generator_rerandomization
    }
    
    tau_hats = []
    for i in range(iterNum):
        Ws = Ws_generator_equal_size(Xs)
        df = Ys_generator(Xs, Ws)
        tau_hats.append(tau_hat_calculator(df))

    tau_hats_method = []
    if method in methodset:
        for i in range(iterNum):
            Ws = methodset[method](*args)
            df = Ys_generator(Xs, Ws)
            tau_hats_method.append(tau_hat_calculator(df))
    else:
        print('Function not found') 

    # boxplot
    # sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(18, 6))

    ax = plt.boxplot([tau_hats, tau_hats_method])
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight='bold')
    plt.xticks([1, 2], ['tau_hat', 'tau_hat_{0:s}'.format(method)])
    plt.title('Boxplot of tau_hat')

    plt.axhline(y=3, color='blue', linestyle='--')
    # plt.axhline(y=1, color='blue', linestyle='--', linewidth = 0.7)

    # add axis label
    plt.ylabel(r"$\mathbf{\hat{\tau}}$", rotation=0,labelpad=10,fontsize=15, color='darkred')

    print('----------------------------')
    print('The variance of tau_hat:{0:.3f}'.format(np.var(tau_hats)))
    print('The variance of tau_hat_{0:s}:{1:.3f}'.format(method, np.var(tau_hats_method)))


def Xs_clarify(df, clarify_crition):

    df_new = df.copy()
    for key_item, key_value in clarify_crition.items():
        df_new[key_item] = df_new[key_item].apply(lambda x: '>{0:.2f}'.format(key_value) if x > key_value else '<{0:.2f}'.format(key_value))
    
    return df_new

def Ws_generator_stratified(Xs, clarify_crition):

    df = Xs.copy()
    df = Xs_clarify(df, clarify_crition)
    stratify_cols = list(clarify_crition.keys())
    
    # Define the StratifiedShuffleSplit object
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    
    # Stratified random split
    for train_index, test_index in stratified_split.split(df, df[stratify_cols]):
        Ws = [0] * len(df)
        for index in train_index:
            Ws[index] = 1
    return np.array(Ws)
        

def Ws_generator_block(Xs, num_blocks=2):
    # 假设df是您的dataframe，其中每行是一个样本。

    # 定义样本总数
    num_samples = len(Xs)

    # 定义块的数量（为简便起见，这里假设每个块的大小相同）
    block_size = num_samples // num_blocks

    # 创建块分配列表
    block_assignment = [1] * block_size + [0] * block_size

    # 如果样本总数是奇数，我们需要在其中一个块中额外添加一个样本
    if num_samples % num_blocks != 0:
        block_assignment.append(np.random.choice([0, 1]))

    # 随机打乱块分配顺序
    np.random.shuffle(block_assignment)

    # 需要的话，转换为DataFrame
    # randomized_df = pd.DataFrame(block_assignment, columns=['Group Assignment'])

    return np.array(block_assignment)






