# ab_test_alipay.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as sp
from scipy.stats import norm

def main():
    # 1. 数据加载与预处理
    print("正在加载数据...")
    data = pd.read_csv('D:/PycharmProjects/working/code/ABtest/effect_tb.csv', header=None)
    data.columns = ["dt", "user_id", "label", "dmp_id"]
    data = data.drop(columns="dt")
    
    # 2. 数据清洗
    print("\n正在清洗数据...")
    print(f"原始数据量: {len(data)} 条")
    data = data.drop_duplicates()
    print(f"去重后数据量: {len(data)} 条")
    
    # 3. 数据质量检查
    print("\n数据质量检查:")
    
    # 空值检查（修复后）
    print("\n空值统计:")
    print(data.isnull().sum())
    
    # 异常值检查
    print("\n异常值检查:")
    print(data.pivot_table(index="dmp_id", columns="label", 
                         values="user_id", aggfunc="count", margins=True))
    
    # 4. 样本容量验证
    print("\n样本容量验证:")
    min_sample_size = 2167
    sample_sizes = data["dmp_id"].value_counts()
    print(f"策略一样本量: {sample_sizes[2]} | 是否达标: {sample_sizes[2] > min_sample_size}")
    print(f"策略二样本量: {sample_sizes[3]} | 是否达标: {sample_sizes[3] > min_sample_size}")
    
    # 5. 点击率分析
    print("\n各策略点击率:")
    for dmp in [1, 2, 3]:
        group = data[data["dmp_id"] == dmp]
        rate = group["label"].mean()
        print(f"策略{dmp} ({len(group)} 用户): {rate:.2%}")

    # 6. 假设检验（策略二）
    print("\n正在进行策略二假设检验...")
    # 定义样本组
    control = data[data["dmp_id"] == 1]
    treatment = data[data["dmp_id"] == 3]
    
    # 计算参数
    n_old, n_new = len(control), len(treatment)
    c_old, c_new = control["label"].sum(), treatment["label"].sum()
    
    # 方法一：手动计算
    r_old = c_old / n_old
    r_new = c_new / n_new
    r = (c_old + c_new) / (n_old + n_new)
    z = (r_old - r_new) / np.sqrt(r * (1 - r) * (1/n_old + 1/n_new))
    z_alpha = norm.ppf(0.05)
    
    # 方法二：statsmodels
    z_score, p_value = sp.proportions_ztest(
        [c_old, c_new], 
        [n_old, n_new], 
        alternative="smaller"
    )
    
    # 7. 结果输出
    print("\n假设检验结果:")
    print(f"手动计算 Z值: {z:.2f}")
    print(f"函数计算 Z值: {z_score:.2f}")
    print(f"P值: {p_value:.10f}")
    print(f"临界Z值: {z_alpha:.2f}")
    
    # 8. 结论判断
    print("\n最终结论:")
    if z_score < z_alpha and p_value < 0.05:
        print("► 拒绝原假设：策略二显著提升了广告点击率")
        improvement = (r_new - r_old)/r_old
        print(f"► 点击率提升幅度: {improvement:.1%}")
    else:
        print("► 保留原假设：策略二未产生显著效果")

    # 9. 保存处理结果
    data.to_csv("D:/PycharmProjects/working/code/ABtest/processed_data.csv", index=False)
    print("\n数据处理结果已保存至 D:/PycharmProjects/working/code/ABtest/processed_data.csv")

if __name__ == "__main__":
    main()