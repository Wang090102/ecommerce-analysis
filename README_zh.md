# 电商用户行为与RMF价值分析

## 项目背景

假设你是一家电商公司的数据分析师，拿到了近半年的用户交易数据和用户基本信息数据。你的目标是：

1.清洗原始脏数据。

2.分析整体销售趋势。

3.使用 RFM模型（最近一次消费 Recency、消费频率 Frequency、消费金额 Monetary）对用户进行分层。

4.找出高价值客户并提出运营建议。

## 数据集生成

生成用户基本信息表（包含脏数据：年龄和性别缺失、年龄异常）、交易流水表（包含脏数据：金额缺失、重复订单）

生成user.csv和transaction.csv文件

## 第一阶段--数据加载和清洗

对用户表和交易表清洗：删除重复值、填充缺失值、替换异常值

生成clean_user.csv和clean_transaction.csv文件

## 第二阶段--探索性数据分析

利用Pandas的分组聚合（groupby）、透视表（pivot_table）和滚动窗口（rolling）发现数据规律。

<img width="376" height="217" alt="屏幕截图 2026-05-10 214811" src="https://github.com/user-attachments/assets/9bb9b6c6-ee20-407e-bd96-3ff4bc486a28" />
<img width="517" height="202" alt="屏幕截图 2026-05-10 215149" src="https://github.com/user-attachments/assets/d4857d49-8ea6-4a76-91ea-066e17a18f60" />
<img width="386" height="202" alt="屏幕截图 2026-05-10 215202" src="https://github.com/user-attachments/assets/9df2496d-eca8-444d-95d4-bcc2f73177a5" />








