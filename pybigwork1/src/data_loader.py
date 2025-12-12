import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path, save_path=None):
    """
    加载数据，并自动执行详细的统计分析与智能清洗。
    在加载过程中会直接输出数据的健康状况报告。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    print(f"🔄 [ETL流程启动] 正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path)

    # ==========================================
    # 📊 第一阶段：数据体检 (统计原始问题)
    # ==========================================
    initial_rows = len(df)
    print(f"\n📊 [1. 数据体检] 初始规模: {initial_rows} 行, {len(df.columns)} 列")
    
    # 1.1 检查重复
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   ⚠️ 发现重复行: {duplicates} 条 (将在清洗阶段删除)")
    else:
        print("   ✅ 无重复行")

    # 1.2 检查缺失值 (只列出 Top 5)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print(f"   ⚠️ 发现缺失值 (Top 5):")
        for col, count in missing.head(5).items():
            print(f"      - {col}: 缺失 {count} 条 (占比 {count/initial_rows:.1%})")
    else:
        print("   ✅ 无缺失值")

    # 1.3 检查逻辑异常 (别墅没地)
    if 'Landsize' in df.columns and 'Type' in df.columns:
        zero_land_houses = df[(df['Type'] == 'h') & (df['Landsize'] == 0)]
        if len(zero_land_houses) > 0:
            print(f"   ⚠️ 发现逻辑异常: {len(zero_land_houses)} 套别墅(House) 土地面积登记为 0")

    # ==========================================
    # 🧹 第二阶段：智能清洗 (修复问题)
    # ==========================================
    print(f"\n🧹 [2. 执行清洗] 开始修复...")
    
    # 2.1 基础操作
    df = df.drop_duplicates()
    
    # 删掉没有房价的 (Target)
    if 'Price' in df.columns:
        missing_price = df['Price'].isnull().sum()
        if missing_price > 0:
            df = df.dropna(subset=['Price'])
            print(f"      🗑️ 已删除 {missing_price} 条缺失房价的数据")

    # 2.2 智能填补
    # 车位 (Car)
    if df['Car'].isnull().sum() > 0:
        df['Car'] = df['Car'].fillna(0)
        print("      🔧 [Car] 缺失值 -> 已填补为 0")
    
    # 建筑面积 (BuildingArea)
    if 'BuildingArea' in df.columns and df['BuildingArea'].isnull().sum() > 0:
        median_area = df['BuildingArea'].median()
        df['BuildingArea'] = df['BuildingArea'].fillna(median_area)
        print(f"      🔧 [BuildingArea] 缺失值 -> 已填补为中位数 {median_area:.1f}")

    # 建成年份 (YearBuilt)
    if 'YearBuilt' in df.columns and df['YearBuilt'].isnull().sum() > 0:
        mode_year = df['YearBuilt'].mode()[0]
        df['YearBuilt'] = df['YearBuilt'].fillna(mode_year)
        print(f"      🔧 [YearBuilt] 缺失值 -> 已填补为众数 {int(mode_year)}")
        
    # 行政区 (CouncilArea)
    if 'CouncilArea' in df.columns and df['CouncilArea'].isnull().sum() > 0:
        df['CouncilArea'] = df['CouncilArea'].fillna('Unknown')
        print("      🔧 [CouncilArea] 缺失值 -> 已标记为 'Unknown'")

    # 2.3 修复逻辑异常 (Landsize=0 for House)
    if 'Landsize' in df.columns and 'Type' in df.columns:
        mask = (df['Type'] == 'h') & (df['Landsize'] == 0)
        if mask.sum() > 0:
            house_median = df[(df['Type'] == 'h') & (df['Landsize'] > 0)]['Landsize'].median()
            df.loc[mask, 'Landsize'] = house_median
            print(f"      🔧 [Landsize] 修复 {mask.sum()} 套异常别墅数据 -> 已修正为中位数 {house_median:.1f}")

    # ==========================================
    # ✅ 第三阶段：结束
    # ==========================================

    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"💾 清洗后的数据已保存至: {save_path}")
        
    return df