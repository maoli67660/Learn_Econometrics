# 📖 第五章 · 模型诊断与残差分析（Model Diagnostics & Residual Analysis）

---

## 🎯 学习目标

- 理解残差与误差项的区别
- 掌握诊断异方差性、自相关、正态性、异常值的方法
- 学会画残差图，做 Breusch-Pagan / White 检验
- 知道何时使用稳健标准误（Robust SE）

---

## 🔹 5.1 什么是残差？什么是误差项？

| 项目       | 记号           | 是否可观测 | 含义 |
|------------|----------------|------------|------|
| 误差项     | \( \varepsilon_i \) | ❌ 不可观测 | 理论上 Y 与 X 无关的其他影响因素 |
| 残差       | \( \hat{\varepsilon}_i = Y_i - \hat{Y}_i \) | ✅ 可观测 | 实际值与预测值的差，用于估计误差项的性质 |

🧠 **残差是误差项的估计值，是模型“健康状况”的X光片。**

---

## 🔹 5.2 异方差性（Heteroskedasticity）

### ✅ 定义：
> 模型理想假设：误差项方差是常数  
> 异方差性：误差项的方差 **随 X 变化而变化**

---

### 🎯 举例说明：

在预测“收入 = β₀ + β₁ × 教育 + ε”的模型中：

- 若教育年限越高，收入波动也越大（创业 / 大厂 vs 平凡上班族）  
  → 就是典型的异方差性（残差呈“漏斗形”）

---

### ✅ 检测方法一：残差图

- 画 \( \hat{\varepsilon}_i \) 对 X 或 \( \hat{Y}_i \) 的散点图
- 若图像呈“漏斗形”，表示残差波动变大 → 存在异方差

#### 示例图（漏斗形 vs 随机）：

✅ 同方差性：

```
        x x x x x x
       x x x x x x x
      x x x x x x x x
```

❌ 异方差性：

```
        x
       x x
      x   x
     x     x
    x       x
```

---

### ✅ 检测方法二：Breusch-Pagan 检验（BP Test）

1. 拟合原模型，计算残差
2. 对残差平方回归：  
   \( \hat{ε}_i^2 = α_0 + α_1X_i + v_i \)
3. 检查 R²，计算 LM 统计量：  
   \( LM = n × R² \)
4. 与卡方临界值比较，或查看 p 值

```python
from statsmodels.stats.diagnostic import het_breuschpagan
het_breuschpagan(model.resid, model.model.exog)
```

📌 若 **p < 0.05** → 拒绝同方差性 → 存在异方差

---

### ✅ 检测方法三：White 检验

- 更灵活，允许检测平方项和交互项引起的异方差

```python
from statsmodels.stats.diagnostic import het_white
het_white(model.resid, model.model.exog)
```

---

### ✅ 处理方法：

- 使用 **稳健标准误（robust standard errors）**：
  ```python
  model_robust = model.get_robustcov_results()
  ```
- 或对 Y 做对数变换，如 `log(income)` 来“压缩”方差差异

---

## 🔹 5.3 自相关（Autocorrelation）

> 残差项之间存在规律性（比如时间序列数据中的“惯性”）

### 🔍 检测方法：

- **Durbin-Watson 检验**：值在 [0, 4]，接近 2 表示无自相关
- 残差 vs 时间图 / 自相关函数（ACF）图

---

### ✅ 解决方法：

- 加入滞后项（如 Yₜ₋₁）
- 使用时间序列模型（ARIMA）
- 使用 **Newey-West** 稳健标准误，处理异方差 + 自相关

---

## 🔹 5.4 正态性检验

### ✅ 为什么重要？
- 小样本下 t 检验、F 检验成立依赖误差项正态分布

---

### ✅ 检测方法：

- QQ 图：是否落在直线上？
- Jarque-Bera 检验：

```python
from scipy.stats import jarque_bera
jarque_bera(model.resid)
```

---

### ✅ 处理方法：

- 对 Y 做 log、sqrt、Box-Cox 等变换
- 提高样本容量（中心极限定理）

---

## 🔹 5.5 异常值和高杠杆点

| 类型       | 特征                     | 危害 |
|------------|--------------------------|------|
| 异常值     | Y 值远离群体             | 拖歪回归线 |
| 高杠杆点   | X 值远离中心区域         | 控制模型斜率 |

---

### ✅ 检测方法：

- 残差图
- 杠杆值（leverage）：
  ```python
  influence = model.get_influence()
  leverage = influence.hat_matrix_diag
  ```
- Cook’s Distance：若 D > 1，说明影响大

---

### ✅ 处理方法：

- 检查数据是否错误
- 报告时使用全样本与剔除后的结果对比
- 使用 **鲁棒回归（Robust Regression）**

---

## 🔹 5.6 稳健标准误（Robust Standard Errors）

即使存在异方差/自相关，我们也能正确进行 t 检验、F 检验。

### ✅ Python 示例：

```python
model_robust = model.get_robustcov_results()
model_robust.summary()
```

---

## 📝 小节总结（Summary）

| 问题类型   | 检测方法                           | 是否严重看啥 | 常见处理 |
|------------|------------------------------------|----------------|-----------|
| 异方差性   | 残差图、BP检验、White 检验         | 残差呈漏斗、p<0.05 | 使用 robust SE / log 变换 |
| 自相关     | Durbin-Watson、ACF图               | DW 明显≠2     | Newey-West / 滞后项 |
| 正态性     | QQ图、Jarque-Bera检验               | 偏态严重 / 小样本 | log 变换 / 增样本 |
| 异常点     | 杠杆值、Cook's D、残差图            | D>1 或极端点 | 删点 / 鲁棒估计 |

---

## 📦 Takeaway 知识点携带包

- **误差项 ≠ 残差**，但我们靠残差来“诊断误差”
- 残差图是最直观的诊断方式
- 同方差假设不成立 → 标准误、t 检验不可信
- 异常值和高杠杆点可能“拖歪整条回归线”
- 稳健标准误 = 实务研究的安全带

---

📌 第六章预告：我们将学习如何使用**虚拟变量（Dummy Variables）**来处理分类变量，如性别、是否毕业、地区等，构建更灵活的回归模型！

