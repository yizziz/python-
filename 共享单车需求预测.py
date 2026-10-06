# -*- coding: utf-8 -*-

# 导包

import os
import sys
import tempfile

_ASCII_TMP = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "Temp", "ml_joblib")
os.makedirs(_ASCII_TMP, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = _ASCII_TMP
tempfile.tempdir = _ASCII_TMP

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # 无界面后端，直接保存图片（不弹窗）
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 0. 全局配置

DATA_PATH = "BikeData.csv"      # 数据集文件名（与脚本放在同一目录）
OUTPUT_DIR = "output"           # 输出目录：预测结果 + 图表
TEST_SIZE = 0.30                # 测试集比例（题目要求切分 30%）
RANDOM_STATE = 42               # 随机种子，保证结果可复现

# 中文字体配置（解决 matplotlib 中文乱码问题）

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 第一步：数据加载与质量清洗

def load_and_clean(path):

    df = pd.read_csv(path)

    # 1) 统一类别拼写
    df["Seasons"] = df["Seasons"].str.strip().str.capitalize()
    df["Functioning Day"] = df["Functioning Day"].str.strip().replace("Y", "Yes")

    print("清洗前样本数：", len(df))

    # 2) 删除完全重复的行
    df = df.drop_duplicates().reset_index(drop=True)
    print("删除完全重复行后：", len(df))

    # 3) 删除目标值异常（-1）的行
    df = df[df["Rented Bike Count"] >= 0].reset_index(drop=True)
    print("删除目标值异常行后：", len(df))

    # 4) 删除系统未运营（Functioning Day = No）的行
    df = df[df["Functioning Day"] == "Yes"].reset_index(drop=True)
    print("删除未运营时段后：", len(df))

    return df



# 第二步：探索性数据分析（EDA）

def exploratory_analysis(df):

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) 目标变量分布
    axes[0, 0].hist(df["Rented Bike Count"], bins=50, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title("Rented Bike Count 分布")
    axes[0, 0].set_xlabel("每小时租赁量")
    axes[0, 0].set_ylabel("频数")

    # (2) 一天内各小时的租赁规律
    hourly = df.groupby("Hour")["Rented Bike Count"].mean()
    axes[0, 1].plot(hourly.index, hourly.values, marker="o", color="#55A868")
    axes[0, 1].set_title("各小时平均租赁量")
    axes[0, 1].set_xlabel("小时")
    axes[0, 1].set_ylabel("平均租赁量")
    axes[0, 1].grid(alpha=0.3)

    # (3) 各季节的租赁规律
    seasonal = df.groupby("Seasons")["Rented Bike Count"].mean().reindex(
        ["Spring", "Summer", "Autumn", "Winter"]
    )
    axes[1, 0].bar(seasonal.index, seasonal.values, color="#C44E52")
    axes[1, 0].set_title("各季节平均租赁量")
    axes[1, 0].set_ylabel("平均租赁量")

    # (4) 相关性热力图（数值特征 + 目标）
    num_cols = [
        "Rented Bike Count", "Hour", "Temperature(°C)", "Humidity(%)",
        "Wind speed (m/s)", "Visibility (10m)", "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)",
    ]
    corr = df[num_cols].corr()
    im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(num_cols)), num_cols, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_yticks(range(len(num_cols)), num_cols, fontsize=8)
    axes[1, 1].set_title("特征相关性热力图")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    eda_path = os.path.join(OUTPUT_DIR, "eda.png")
    plt.savefig(eda_path, dpi=120)
    plt.close()
    print("EDA 图表已保存：", eda_path)



# 第三步：特征工程

def build_features(df):

    df = df.copy()

    # 1) 解析日期并拆解
    dt = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Weekday"] = dt.dt.weekday                     # 0=周一 ... 6=周日
    df["IsWeekend"] = (dt.dt.weekday >= 5).astype(int)  # 周六日标记为 1

    # 2) 小时周期编码
    df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # 3) 类别编码（Seasons / Holiday）
    season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
    df["SeasonCode"] = df["Seasons"].map(season_map)
    df["HolidayCode"] = (df["Holiday"] == "Holiday").astype(int)

    # 4) 最终特征列：气象特征 + 时间特征 + 类别编码
    feature_cols = [
        "Hour", "HourSin", "HourCos",
        "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
        "Visibility (10m)", "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)",
        "Month", "Day", "Weekday", "IsWeekend",
        "SeasonCode", "HolidayCode",
    ]
    X = df[feature_cols].astype(float)
    y = df["Rented Bike Count"].astype(float)
    return X, y, feature_cols



# 第四步：模型训练、对比与评估

def train_and_evaluate(X, y, feature_cols):

    # ---- 1) 切分训练/测试集（70% / 30%）----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")

    # ---- 2) 候选模型（交叉验证对比）----
    models = {
        "线性回归 LinearRegression": LinearRegression(),
        "随机森林 RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "梯度提升 GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    print("\n===== 5 折交叉验证对比（RMSE，越小越好）=====")
    cv_results = {}
    for name, model in models.items():
        # 线性回归对量纲敏感，先做标准化
        if isinstance(model, LinearRegression):
            from sklearn.pipeline import make_pipeline
            model = make_pipeline(StandardScaler(), model)
        scores = -cross_val_score(model, X_train, y_train, cv=5,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_results[name] = scores.mean()
        print(f"{name:28s}  CV-RMSE = {scores.mean():8.2f} (±{scores.std():.2f})")

    # 选出交叉验证最优模型
    best_name = min(cv_results, key=cv_results.get)
    print(f"\n交叉验证最优模型：{best_name}")

    # ---- 3) 对最优模型做超参数网格搜索（轻量调优）----
    final_model = None
    if "随机森林" in best_name:
        # 随机森林调优：树的数量与最大深度
        param_grid = {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]}
        base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                          n_jobs=-1)
        gs.fit(X_train, y_train)
        print("随机森林最优参数：", gs.best_params_)
        final_model = gs.best_estimator_
    elif "梯度提升" in best_name:
        # 梯度提升调优：学习率与树的数量
        param_grid = {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1]}
        base = GradientBoostingRegressor(random_state=RANDOM_STATE)
        gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                          n_jobs=-1)
        gs.fit(X_train, y_train)
        print("梯度提升最优参数：", gs.best_params_)
        final_model = gs.best_estimator_
    else:
        # 线性回归：直接使用（带标准化管道）
        from sklearn.pipeline import make_pipeline
        final_model = make_pipeline(StandardScaler(), LinearRegression())

    # ---- 4) 在 30% 测试集上评估 ----
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n===== 测试集最终评估（30% 测试集）=====")
    print(f"MSE  = {mse:10.2f}")
    print(f"RMSE = {rmse:10.2f}")
    print(f"MAE  = {mae:10.2f}")
    print(f"R^2  = {r2:10.4f}")

    # ---- 5) 保存测试集预测结果 ----
    result_df = pd.DataFrame({
        "Rented Bike Count（真实值）": y_test.values,
        "Rented Bike Count（预测值）": np.round(y_pred, 1),
        "绝对误差": np.abs(y_test.values - y_pred).round(1),
    })
    result_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print("测试集预测结果已保存：", result_path)

    # ---- 6) 输出图表：特征重要性 + 预测值 vs 真实值 ----
    plot_feature_importance(final_model, feature_cols)
    plot_prediction_scatter(y_test.values, y_pred)

    return final_model, mse, rmse, mae, r2



# 可视化辅助函数：特征重要性 & 预测对比

def plot_feature_importance(model, feature_cols):
    """对树模型输出特征重要性柱状图（线性模型则跳过）。"""
    if not hasattr(model, "feature_importances_"):
        print("当前模型不支持特征重要性，跳过该图。")
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(9, 6))
    imp.plot.barh(color="#4C72B0")
    plt.title("特征重要性（最优模型）")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print("特征重要性图已保存：", path)


def plot_prediction_scatter(y_true, y_pred):
    """预测值 vs 真实值散点图 + 残差分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 散点图（附 y=x 参考线）
    axes[0].scatter(y_true, y_pred, s=6, alpha=0.4, color="#55A868")
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    axes[0].plot(lim, lim, "r--", linewidth=1.5, label="y = x（理想线）")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].set_title("测试集：预测值 vs 真实值")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 残差分布直方图
    residual = y_true - y_pred
    axes[1].hist(residual, bins=50, color="#C44E52", edgecolor="white")
    axes[1].set_xlabel("残差（真实值 - 预测值）")
    axes[1].set_ylabel("频数")
    axes[1].set_title("测试集残差分布")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "prediction_result.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print("预测对比图已保存：", path)





if __name__ == "__main__":
    print("=" * 60)
    print("城市共享单车需求预测")
    print("=" * 60)

    # 第一步：加载与清洗
    df = load_and_clean(DATA_PATH)
    print("清洗后样本数：", len(df))

    # 第二步：EDA（若不需要可视化，可注释掉下一行）
    exploratory_analysis(df)

    # 第三步：特征工程
    X, y, feature_cols = build_features(df)
    print("特征数量：", len(feature_cols))

    # 第四步：建模、调优与评估
    train_and_evaluate(X, y, feature_cols)

    print("\n全部完成！产物见 output/ 目录。")
