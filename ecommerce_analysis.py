import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings

# 忽略 Pandas 的链式赋用警告，保持输出整洁
warnings.filterwarnings('ignore')

# ==========================================
# 模块 0: 数据生成（模拟真实业务中的脏数据）
# ==========================================
def generate_dummy_data():
    print("=" * 50)
    print("🚀 正在生成模拟数据...")
    np.random.seed(42)

    # 1. 生成用户基本信息表 (包含缺失值、异常年龄)
    user_ids = [f"U{str(i).zfill(4)}" for i in range(1, 101)]
    genders = np.random.choice(['M', 'F', np.nan], size=100, p=[0.45, 0.45, 0.10])
    ages = np.random.randint(18, 60, size=100).astype(float)
    ages[np.random.choice(100, 5)] = -1   # 制造异常年龄
    ages[np.random.choice(100, 10)] = np.nan # 制造缺失年龄
    cities = np.random.choice(['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu'], size=100)

    users_df = pd.DataFrame({'UserID': user_ids, 'Gender': genders, 'Age': ages, 'City': cities})

    # 2. 生成交易流水表 (包含缺失金额、重复订单)
    trans_data = []
    end_date = datetime(2023, 12, 31)
    start_date = datetime(2023, 0o7, 0o1)

    for i in range(1000):
        uid = random.choice(user_ids)
        trans_date = start_date + timedelta(days=random.randint(0, 184))
        amount = round(random.uniform(10, 2000), 2)
        category = random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'])
        trans_data.append([f"T{str(i).zfill(5)}", uid, trans_date, amount, category])

    trans_df = pd.DataFrame(trans_data, columns=['OrderID', 'UserID', 'OrderDate', 'Amount', 'Category'])

    # 制造脏数据
    trans_df.loc[np.random.choice(1000, 30, replace=False), 'Amount'] = np.nan  # 金额缺失
    trans_df = pd.concat([trans_df, trans_df.iloc[[0, 1]]], ignore_index=True) # 制造重复行

    print("✅ 数据生成完毕！")
    return users_df, trans_df


# ==========================================
# 模块 1: 数据清洗
# ==========================================
def clean_data(df_trans, df_users):
    print("\n" + "=" * 50)
    print("🧹 正在进行数据清洗...")
    
    # --- 交易表清洗 ---
    # 1. 处理重复值
    print(f"  交易记录删除前: {len(df_trans)} 行")
    df_trans.drop_duplicates(inplace=True)
    print(f"  交易记录删除后: {len(df_trans)} 行")

    # 2. 处理缺失值 (金额缺失用中位数填充)
    median_amount = df_trans['Amount'].median()
    df_trans['Amount'].fillna(median_amount, inplace=True)

    # 3. 转换时间格式 & 提取特征
    df_trans['OrderDate'] = pd.to_datetime(df_trans['OrderDate'])
    df_trans['YearMonth'] = df_trans['OrderDate'].dt.to_period('M')
    df_trans['DayOfWeek'] = df_trans['OrderDate'].dt.day_name()

    # --- 用户表清洗 ---
    # 1. 处理异常年龄 (-1 替换为 NaN)
    df_users['Age'] = df_users['Age'].apply(lambda x: np.nan if x < 0 else x)
    # 2. 按城市均值填充缺失年龄 (使用 transform 保证索引对齐)
    city_mean_age = df_users.groupby('City')['Age'].transform('mean')
    df_users['Age'].fillna(city_mean_age, inplace=True)
    # 3. 性别缺失填充
    df_users['Gender'].fillna('Unknown', inplace=True)
    
    print("✅ 数据清洗完毕！")
    return df_trans, df_users


# ==========================================
# 模块 2: 探索性分析 (EDA)
# ==========================================
def exploratory_analysis(df_trans):
    print("\n" + "=" * 50)
    print("📊 正在进行探索性数据分析...")
    
    # 1. 按月统计总销售额和订单量
    monthly_stats = df_trans.groupby('YearMonth').agg(
        Total_Amount=('Amount', 'sum'),
        Order_Count=('OrderID', 'count')
    ).reset_index()
    print("\n--- 每月销售统计 ---")
    print(monthly_stats)

    # 2. 品类透视表
    category_pivot = pd.pivot_table(
        df_trans, values='Amount', index='Category', 
        aggfunc=['sum', 'mean', 'count']
    )
    category_pivot.columns = ['总销售额', '平均客单价', '订单数']
    category_pivot = category_pivot.sort_values(by='总销售额', ascending=False)
    print("\n--- 品类销售透视表 ---")
    print(category_pivot)

    # 3. 7日移动平均销售额
    daily_sales = df_trans.set_index('OrderDate').resample('D')['Amount'].sum()
    daily_sales_ma7 = daily_sales.rolling(window=7).mean()
    print("\n--- 最近7天的7日移动平均销售额 ---")
    print(daily_sales_ma7.tail(7))
    
    print("✅ 探索性分析完毕！")


# ==========================================
# 模块 3: RFM 用户价值分层
# ==========================================
def rfm_analysis(df_trans, df_users):
    print("\n" + "=" * 50)
    print("🎯 正在进行 RFM 用户价值分层...")
    
    current_date = df_trans['OrderDate'].max() + timedelta(days=1)

    # 计算 RFM 原始值
    rfm_df = df_trans.groupby('UserID').agg(
        Recency=('OrderDate', lambda x: (current_date - x.max()).days),
        Frequency=('OrderID', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()

    # RFM 五分位数打分
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    # Frequency 常存在偏态，使用 rank(method='first') 避免分箱报错
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])

    # 计算总分
    rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)

    # 用户层级划分
    def assign_label(score):
        if score >= 13: return '重要价值客户'
        elif score >= 10: return '重要发展客户'
        elif score >= 7: return '重要保持客户'
        else:
            return '一般客户'

    rfm_df['User_Label'] = rfm_df['RFM_Score'].apply(assign_label)
    
    print("\n--- 用户分层分布 ---")
    print(rfm_df['User_Label'].value_counts())

    # 合并用户信息并做交叉表分析
    final_df = pd.merge(rfm_df, df_users, on='UserID', how='left')
    
    city_label_pct = pd.crosstab(final_df['City'], final_df['User_Label']).apply(
        lambda x: round(x / x.sum(), 2), axis=1
    )
    print("\n--- 各城市用户层级占比 ---")
    print(city_label_pct)

    # 找出沉睡高净值客户
    sleeping_vip = final_df[(final_df['R_Score'].astype(int) <= 2) & (final_df['M_Score'].astype(int) >= 4)]
    print("\n--- 需要唤醒的沉睡高净值客户 (前5名) ---")
    print(sleeping_vip[['UserID', 'Recency', 'Monetary', 'City', 'Gender']].head())

    print("✅ RFM 分析完毕！")
    return final_df


# ==========================================
# 主函数入口
# ==========================================
def main():
    # 0. 生成数据
    users_df, trans_df = generate_dummy_data()
    
    # 1. 数据清洗
    clean_trans, clean_users = clean_data(trans_df, users_df)
    
    # 2. 探索性分析
    exploratory_analysis(clean_trans)
    
    # 3. RFM 分析
    final_report = rfm_analysis(clean_trans, clean_users)
    
    # 4. 导出最终数据
    final_report.to_csv('user_rfm_analysis_report.csv', index=False, encoding='utf-8-sig')
    print("\n" + "=" * 50)
    print("💾 分析报告已成功导出为 'user_rfm_analysis_report.csv'")

# 只有当脚本被直接运行时才执行主函数
if __name__ == '__main__':
    main()
