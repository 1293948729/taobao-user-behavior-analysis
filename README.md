# 淘宝用户行为分析

基于用户行为日志进行数据清洗、探索分析和特征工程，分析用户活跃、商品热度、类目偏好及购买行为，并构建用户商品对的七日购买标签。

## 安装

使用 Python 3.10。

## 输入数据

CSV 文件名为 `user_behavior.csv`，表头顺序如下：

```text
time,user_id,item_id,item_category,behavior_type
```

| 字段 | 含义 |
| --- | --- |
| `time` | 行为发生时间，格式为 `2025-12-01 10`，精确到小时 |
| `user_id` | 用户编号，按字符串读取 |
| `item_id` | 商品编号，按字符串读取 |
| `item_category` | 商品类目编号 |
| `behavior_type` | 1 浏览、2 收藏、3 加购、4 购买 |


## 使用方法

```powershell
# 检查输入格式和前 1000 行
.\.venv\Scripts\python.exe run_project.py --stage inspect --input ./data/raw/user_behavior.csv.zip

# 清洗数据，生成探索分析汇总和图表
.\.venv\Scripts\python.exe run_project.py --stage eda --input ./data/raw/user_behavior.csv.zip

# 使用已清洗的数据构建特征和标签
.\.venv\Scripts\python.exe run_project.py --stage features

# 检查时间分布，筛选特征并准备训练采样方案
.\.venv\Scripts\python.exe run_project.py --stage prepare
```

只做清洗和质量统计时使用 `--stage audit`；已有完整分析结果时，可用 `--stage charts` 重新绘图。运行参数在 `configs/` 中配置。

## 数据处理与分析

用户和商品编号保留前导零；关键字段无效的记录单独隔离，缺失类目保留并增加标记。同一用户、商品、行为和小时的重复次数跨数据块统计，超过 100 次时隔离整组。用户每小时总行为量和全期行为量的 IQR 仅用于诊断。

探索分析包括行为构成与时间分布、用户活跃与购买频次、首次观察分组留存、商品和类目排名、长尾分布、行为共现及后续购买关系。

日志仅精确到小时，不推断小时内的行为先后；购买事件数不作为订单数，购买与浏览的事件比不作为转化概率。

## 特征工程

样本以 `user_id + item_id + cutoff` 为唯一键。候选取自历史七天内有交互的用户商品对，特征使用 `[cutoff-7天, cutoff)`，标签判断 `[cutoff, cutoff+7天)` 内是否发生购买。

按时间划分 train、validation、test，三个标签窗口相邻且不重叠；另生成 latest 无标签样本。数据至少需要 28 个完整日，每套特征均从对应历史窗口独立计算，保留自然正负比例。

| 分组 | 特征数 | 主要内容 |
| --- | ---: | --- |
| 用户商品交互 | 11 | 行为计数、近期行为、活跃天数、最近行为距离 |
| 用户活跃与广度 | 11 | 行为量、交互商品数、类目数、活跃天数 |
| 商品热度 | 9 | 行为量、交互用户数、近期热度 |
| 类目热度 | 7 | 行为量、交互用户数、商品数 |
| 偏好占比与辅助标记 | 10 | 类目偏好、交互占比、事件比、缺失标记、截点星期 |

## 训练数据准备

保留完整特征表，另按固定字段清单生成建模用表。筛选只使用完整训练集，不使用标签或验证、测试数据确定筛选规则。检查常量、完全重复、行为总量关系及同层特征的秩相关。时间距离、占比、事件比、活跃天数和跨层关系保留业务区分，高相关只作提示。

训练负采样保留全部正例，按固定种子抽取负例。默认负正比上限为100:1，同时统计20:1、50:1和100:1的样本规模与覆盖。完整自然训练数据另有类别权重方案，两种方案不默认叠加。验证集和测试集保持自然比例，latest不含标签。

参数在`configs/project.toml`的`preparation`中配置。完整宽表保存在`artifacts/data/features`，筛选后的数据在`artifacts/data/prepared`，统计和字段清单在`artifacts/outputs/preparation`。这些方案尚未通过模型效果对比。

## 项目结构

```text
src/taobao_user_behavior/
  data/             数据读取、标准化和清洗
  analysis/         SQL 聚合与探索分析
  visualization/    图表绘制
  features/         样本窗口、特征计算和字段说明
  preparation/      时间核验、特征筛选和训练采样
  settings.py       参数与路径配置
  storage.py        文件读写
scripts/            代码检查和结果校验
tests/             单元测试
configs/            项目配置
run_project.py      运行入口
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_project.py

# 已生成准备结果后，独立核对筛选投影、采样和统计
.\.venv\Scripts\python.exe scripts/verify_preparation.py
```
