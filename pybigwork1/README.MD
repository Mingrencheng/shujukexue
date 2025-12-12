# 墨尔本房价预测分析 

## 📖 项目简介
本项目是《数据科学概论》课程大作业。项目旨在通过分析墨尔本房屋交易数据集 (`melb_data.csv`)，探究影响房价的关键因素（如房间数、房屋类型、地理位置、建筑面积等），并构建线性回归模型对房价进行预测。

核心目标：**数据理解、清洗严谨性、可视化分析与结果解释**。

## 📂 目录结构
```text
housing_analysis/
├── data/
│   ├── raw/                 # 原始数据 (melb_data.csv 放在此处)
│   └── processed/           # 清洗后的数据 (代码自动生成)
├── notebooks/
│   └── exploration.ipynb    # 核心分析与建模代码 (Jupyter Notebook)
├── src/                     # 源代码模块 (工程化封装)
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载与智能清洗逻辑
│   ├── features.py          # 特征工程与预处理管道
│   └── visualization.py     # 绘图函数封装
├── reports/
│   ├── figures/             # 生成的分析图表 (自动保存)
│   └── final-report.pdf     # 最终分析报告
├── README.md                # 项目说明文档
└── requirements.txt         # 项目依赖库
```

## 🚀 如何运行

1.  **环境配置**：
    确保安装 Python 3.x，并安装所需依赖：

    ```bash
    pip install -r requirements.txt
    ```

    *(依赖包含: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter)*

2.  **数据准备**：
    请确保原始数据文件 `melb_data.csv` 位于 `data/raw/` 目录下。

3.  **运行分析**：
    打开 `notebooks/exploration.ipynb`，从上到下依次运行所有单元格。代码将自动执行：

      * 加载并清洗数据
      * 生成并保存 4 张 EDA 图表到 `reports/figures/`
      * 训练线性回归模型与基线模型
      * 输出 R² 分数评估结果

## 🛠️ 数据处理与清洗策略

在 `src/data_loader.py` 中实现了详细的清洗逻辑，针对原始数据的质量问题进行了以下修复：

1.  **缺失值处理**：
      * **建筑面积 (BuildingArea)**：缺失约 47%，使用**中位数**填补（避免豪宅极大值影响）。
      * **建成年份 (YearBuilt)**：缺失约 40%，使用**众数**（最常见年份）填补。
      * **车位 (Car)**：缺失值填补为 0。
      * **行政区 (CouncilArea)**：缺失值标记为 'Unknown'。
2.  **异常值修复**：
      * **土地面积 (Landsize)**：修复了 171 条 **别墅 (House)** 土地面积为 0 的错误数据（使用别墅的中位数替换）；对于公寓 (Unit)，保留 0 值（代表无独立地权）。

## 📊 探索性数据分析 (EDA) 结论

项目生成了以下关键图表（位于 `reports/figures/`）：

1.  **房价分布直方图 (`price_distribution.png`)**：

      * 结论：房价呈明显的**右偏分布**，大部分房屋集中在 60万-120万澳元之间，存在少量高价豪宅。

2.  **特征相关性热力图 (`correlation_heatmap.png`)**：

      * 结论：**房间数 (Rooms)**、**卧室数 (Bedroom2)** 与房价的相关性最强。土地面积 (Landsize) 与房价的相关性较弱。

3.  **面积-房价散点图 (`area_price_scatter.png`)**：

      * 结论：建筑面积与房价呈正相关趋势，但存在部分离群点（如面积很大但价格低的房产）。

## 🤖 建模与评估

本项目使用 **Scikit-Learn Pipeline** 构建了端到端的预测模型：

  * **特征工程**：
      * 数值特征：中位数填补缺失值 + StandardScaler 标准化。
      * 类别特征：众数填补缺失值 + OneHotEncoder 独热编码。
  * **模型选择**：
      * **基线模型 (Baseline)**：总是预测均值 (DummyRegressor)。
      * **核心模型**：多元线性回归 (Linear Regression)。

### 评估结果 (示例)

  * **基线模型 R² Score**: 0.0000 (参考线)
  * **线性回归 R² Score**: **0.XXXX** (显著优于基线，说明模型有效解释了房价波动)

## 📝 总结

本项目通过规范的数据科学流程，成功识别了影响墨尔本房价的主要因素。数据清洗阶段的精细化处理（特别是针对 BuildingArea 和 Landsize 的修复）显著提升了数据质量，为模型预测奠定了基础。


