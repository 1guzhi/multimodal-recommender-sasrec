# 多模态商品推荐系统（SASRec + ResNet50）

基于 SASRec 序列模型的多模态商品推荐系统。融合 ResNet50 图像特征与 TF-IDF 文本特征，与仅使用商品 ID 的基线模型对比，验证多模态信息对 Top-N 排序精度的提升。

> 智能推荐系统课程作业（2026 春），数据来自课程 Kaggle 排位赛。

## 技术栈
- Python / PyTorch
- SASRec 序列推荐模型
- ResNet50 预训练模型提取图像特征（去掉分类层，取 2048 维）
- TF-IDF + TruncatedSVD 提取文本特征
- 评估指标：NDCG@20

## 主要功能
- `full_recommender.py`：完整流程——商品图片批量特征提取、文本 TF-IDF 特征、用户交互序列负采样、SASRec 模型训练与评估
- ID-only 与多模态两套模型对比（`model_comparison_results.csv`）
- 超参数扫描：embed_size / num_layers / num_heads / dropout（`hyperparam_*.csv`）
- `报告.docx`：完整实验报告

## 如何运行
```bash
# 依赖：torch、torchvision、pandas、numpy、scikit-learn、seaborn、matplotlib、tqdm
python full_recommender.py
```

代码默认从 `DATA_DIR` 读取 `train.csv / valid.csv / test_users.csv / items.csv`，商品图片位于 `images/` 目录（图片数据体积较大，未纳入本仓库，可从课程数据集获取）。

## 结果
- ID-only vs 多模态 对比结果：`model_comparison_results.csv`
- 各超参数影响：`hyperparameter_summary.csv`、`hyperparam_*.csv`
