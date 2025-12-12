# 将以下代码复制到 src/features.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def make_preprocessor():
    """
    构建并返回预处理管道
    """
    # 数值型特征
    numerical_features = [
        'Rooms', 'Distance', 'Postcode', 'Bedroom2', 
        'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt'
    ]
    
    # 类别型特征
    categorical_features = ['Type', 'Method', 'Regionname']

    # 数值处理：补缺失值 -> 标准化
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 类别处理：补缺失值 -> OneHot编码
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 组合起来
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor