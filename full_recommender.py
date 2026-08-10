import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DATA_DIR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据加载、基本统计
print("\n加载数据...")
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_users_df = pd.read_csv('test_users.csv')
items_df = pd.read_csv('items.csv')

print(f"训练集: {len(train_df)}条交互")
print(f"验证集: {len(valid_df)}个用户")
print(f"测试集: {len(test_users_df)}个用户")
print(f"商品数: {len(items_df)}个")


# 价格清洗 + 标准化
def clean_price(price):
    if pd.isna(price):
        return 0.0
    if isinstance(price, str):
        price = price.replace('$', '').replace(',', '').strip()
        try:
            return float(price)
        except:
            return 0.0
    return float(price)


items_df['price_clean'] = items_df['price'].apply(clean_price)
price_scaler = StandardScaler()
items_df['price_scaled'] = price_scaler.fit_transform(items_df[['price_clean']])

# 过滤无图商品、构建item/user映射
items_df = items_df.dropna(subset=['image_path']).reset_index(drop=True)
print(f"有图片的商品数: {len(items_df)}")

item2idx = {item: idx for idx, item in enumerate(items_df['item_id'])}
idx2item = {idx: item for item, idx in item2idx.items()}
num_items = len(items_df)

train_df = train_df[train_df['item_id'].isin(item2idx.keys())].reset_index(drop=True)
valid_df = valid_df[valid_df['item_id'].isin(item2idx.keys())].reset_index(drop=True)

users = sorted(train_df['user_id'].unique())
user2idx = {u: i for i, u in enumerate(users)}
num_users = len(users)

print(f"有效用户数: {num_users}, 有效商品数: {num_items}")

# 图像transform + ResNet50特征提取器
print("\n提取多模态特征...")
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageExtractor:
    def __init__(self):
        model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(device)

    def extract(self, img_path):
        try:
            img_path_full = os.path.join(DATA_DIR, img_path)
            if not os.path.exists(img_path_full):
                return np.zeros(2048)
            img = Image.open(img_path_full).convert('RGB')
            img_tensor = img_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = self.model(img_tensor).squeeze().cpu().numpy()
            return feat
        except Exception as e:
            return np.zeros(2048)


# 批量提取图像特征
print("提取图像特征中...")
extractor = ImageExtractor()
image_feats = []
for _, row in tqdm(items_df.iterrows(), desc="提取图像特征"):
    feat = extractor.extract(row['image_path'])
    image_feats.append(feat)

image_matrix = np.nan_to_num(np.array(image_feats))
print(f"图像特征维度: {image_matrix.shape}")

# 文本拼接 + TF-IDF
print("提取文本特征中...")
texts = []
for _, row in items_df.iterrows():
    title = str(row['title']) if pd.notna(row['title']) else ""
    desc = str(row['description']) if pd.notna(row['description']) else ""
    texts.append(f"{title} {title} {desc}")

tfidf = TfidfVectorizer(
    max_features=1024,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.8
)
text_matrix = tfidf.fit_transform(texts).toarray()
text_matrix = np.nan_to_num(text_matrix)
print(f"文本特征维度: {text_matrix.shape}")

# PCA降维 + 拼接图像/文本/价格
print("降维处理...")
pca_img = PCA(n_components=128)
image_feats_128 = pca_img.fit_transform(image_matrix)

pca_text = PCA(n_components=128)
text_feats_128 = pca_text.fit_transform(text_matrix)

price_feats = items_df['price_scaled'].values.reshape(-1, 1)
final_features = np.concatenate([
    image_feats_128,
    text_feats_128,
    price_feats
], axis=1)

feature_scaler = StandardScaler()
final_features = feature_scaler.fit_transform(final_features)

item_features = torch.FloatTensor(final_features).to(device)
print(f"商品最终特征维度: {item_features.shape}")
# 构建用户交互序列
print("\n构建用户序列...")
train_df = train_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
user_seq = defaultdict(list)
for uid, iid in zip(train_df['user_id'], train_df['item_id']):
    user_seq[uid].append(item2idx[iid])

train_sequences = []
min_seq_len = 2
max_seq_len = 50

for uid in users:
    seq = user_seq[uid]
    if len(seq) >= min_seq_len:
        for i in range(min_seq_len, min(len(seq), max_seq_len + 1)):
            train_sequences.append((user2idx[uid], tuple(seq[:i])))

print(f"训练样本数: {len(train_sequences)}")


# SASRec Dataset（负采样+序列padding）
class SASRecDataset(Dataset):
    def __init__(self, sequences, num_items, max_len=50, neg_samples=10):
        self.sequences = sequences
        self.num_items = num_items
        self.max_len = max_len
        self.neg_samples = neg_samples

        self.item_pop = defaultdict(int)
        for _, seq in sequences:
            for item in seq:
                self.item_pop[item] += 1
        self.item_pop_array = np.array([self.item_pop.get(i, 1) for i in range(num_items)])
        self.item_pop_array = self.item_pop_array ** 0.75
        self.item_pop_array = self.item_pop_array / self.item_pop_array.sum()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        user, seq = self.sequences[idx]
        seq = list(seq)

        input_seq = seq[:-1]
        target = seq[-1]
        interacted = set(seq)
        neg_items = []
        while len(neg_items) < self.neg_samples:
            candidate = np.random.choice(self.num_items, p=self.item_pop_array)
            if candidate not in interacted:
                neg_items.append(candidate)
        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]
        else:
            pad_len = self.max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq

        return (
            torch.LongTensor(input_seq),
            torch.LongTensor([target]),
            torch.LongTensor(neg_items)
        )


# SASRec 模型（融合多模态特征）
class SASRecModel(nn.Module):
    def __init__(self, num_items, item_features_dim=257, embed_size=256,
                 max_len=50, num_heads=8, num_layers=2, dropout=0.2, use_multimodal=True):
        super().__init__()
        self.num_items = num_items
        self.embed_size = embed_size
        self.max_len = max_len
        self.use_multimodal = use_multimodal

        self.item_embedding = nn.Embedding(num_items + 1, embed_size, padding_idx=0)

        if use_multimodal:
            self.modal_projection = nn.Sequential(
                nn.Linear(item_features_dim, embed_size),
                nn.LayerNorm(embed_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_size, embed_size),
                nn.LayerNorm(embed_size),
            )

        self.pos_embedding = nn.Embedding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layernorm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq, item_features):
        batch_size, seq_len = seq.shape

        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        item_emb = self.item_embedding(seq)

        if self.use_multimodal:
            seq_flat = seq.view(-1)
            mask = seq_flat != 0
            modal_emb = torch.zeros(batch_size * seq_len, self.embed_size, device=seq.device)

            if mask.any():
                valid_indices = torch.where(mask)[0]
                valid_items = seq_flat[valid_indices]
                modal_out = self.modal_projection(item_features[valid_items])
                modal_emb[valid_indices] = modal_out

            modal_emb = modal_emb.view(batch_size, seq_len, -1)
            seq_emb = item_emb + modal_emb + pos_emb
        else:
            seq_emb = item_emb + pos_emb

        seq_emb = self.layernorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device) * float('-inf'), diagonal=1)
        output = self.transformer(seq_emb, mask=causal_mask)

        return output

    def get_scores(self, seq, item_features, target_items, neg_items):
        output = self.forward(seq, item_features)
        user_emb = output[:, -1, :]

        if self.use_multimodal:
            target_emb = self.item_embedding(target_items) + self.modal_projection(item_features[target_items])
            neg_emb = self.item_embedding(neg_items) + self.modal_projection(item_features[neg_items])
        else:
            target_emb = self.item_embedding(target_items)
            neg_emb = self.item_embedding(neg_items)

        pos_scores = (user_emb * target_emb).sum(dim=1)
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(2)).squeeze(2)

        return pos_scores, neg_scores

    def predict_all(self, seq, item_features):
        output = self.forward(seq, item_features)
        user_emb = output[:, -1, :]

        if self.use_multimodal:
            all_item_emb = self.item_embedding.weight[1:] + self.modal_projection(item_features)
        else:
            all_item_emb = self.item_embedding.weight[1:]

        scores = torch.matmul(user_emb, all_item_emb.T)
        return scores


# 训练函数（BPR损失）
def train_epoch(model, train_loader, optimizer, item_features, device):
    model.train()
    total_loss = 0

    for seq, target, neg_items in tqdm(train_loader, desc="Training"):
        seq = seq.to(device)
        target = target.squeeze().to(device)
        neg_items = neg_items.to(device)

        pos_scores, neg_scores = model.get_scores(seq, item_features, target, neg_items)

        pos_scores = pos_scores.unsqueeze(1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# 评估函数（NDCG@20）
def evaluate(model, valid_df, user_seq_train, item_features, max_len=50, K=20):
    model.eval()
    all_ndcg_scores = []

    with torch.no_grad():
        for uid in tqdm(valid_df['user_id'].unique(), desc="Evaluating"):
            valid_items = valid_df[valid_df['user_id'] == uid]['item_id'].values
            true_indices = []
            for item in valid_items:
                if item in item2idx:
                    true_indices.append(item2idx[item])

            if len(true_indices) == 0:
                continue
            if uid not in user_seq_train or len(user_seq_train[uid]) == 0:
                scores = np.ones(num_items)
                filtered_scores = scores
            else:
                seq = user_seq_train[uid]
                if len(seq) <= max_len:
                    input_seq = [0] * (max_len - len(seq)) + seq
                else:
                    input_seq = seq[-max_len:]

                seq_tensor = torch.LongTensor([input_seq]).to(device)
                scores = model.predict_all(seq_tensor, item_features).squeeze(0).cpu().numpy()
                if len(scores) != num_items:
                    full_scores = np.zeros(num_items)
                    min_len = min(len(scores), num_items)
                    full_scores[:min_len] = scores[:min_len]
                    scores = full_scores
                filtered_scores = scores.copy()
                for idx in seq:
                    if idx < num_items:
                        filtered_scores[idx] = -1e9
            top_k_indices = np.argsort(filtered_scores)[::-1][:K]
            dcg = 0.0
            for i, idx in enumerate(top_k_indices):
                if idx in true_indices:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = 0.0
            for i in range(min(K, len(true_indices))):
                idcg += 1.0 / np.log2(i + 2)

            if idcg > 0:
                ndcg = dcg / idcg
                all_ndcg_scores.append(ndcg)

    if len(all_ndcg_scores) == 0:
        return 0.0

    return np.mean(all_ndcg_scores)


# 基础训练 + 生成提交文件
def run_baseline_training():
    print("\n" + "=" * 80)
    print("基础训练模式 - 训练多模态模型并生成提交文件")
    print("=" * 80)
    # 超参数配置
    max_len = 50
    batch_size = 128
    embed_size = 256
    num_epochs = 30
    learning_rate = 0.001
    neg_samples = 10
    num_heads = 4
    num_layers = 2
    dropout = 0.2

    # 创建数据集和数据加载器
    train_dataset = SASRecDataset(train_sequences, num_items, max_len, neg_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建多模态模型
    model = SASRecModel(
        num_items=num_items,
        item_features_dim=257,
        embed_size=embed_size,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_multimodal=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # 训练循环
    best_ndcg = 0.0
    patience = 5
    wait = 0

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, item_features, device)

        if (epoch + 1) % 3 == 0:
            val_ndcg = evaluate(model, valid_df, user_seq, item_features, max_len, K=20)
            print(f"Epoch {epoch + 1:2d}/{num_epochs} | Loss: {loss:.4f} | NDCG@20: {val_ndcg:.4f}")

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                torch.save(model.state_dict(), 'best_model_multimodal.pth')
                print(f"  保存最佳模型 (NDCG@20={best_ndcg:.4f})")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            print(f"Epoch {epoch + 1:2d}/{num_epochs} | Loss: {loss:.4f}")

        scheduler.step()

    print(f"\n最佳验证NDCG@20: {best_ndcg:.4f}")
    print("\n生成提交文件...")
    # 全量数据训练
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)
    full_train_df = full_train_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    full_user_seq = defaultdict(list)
    for uid, iid in zip(full_train_df['user_id'], full_train_df['item_id']):
        if iid in item2idx:
            full_user_seq[uid].append(item2idx[iid])

    full_sequences = []
    for uid in full_user_seq.keys():
        if uid not in user2idx:
            user2idx[uid] = len(user2idx)
        seq = full_user_seq[uid]
        if len(seq) >= 2:
            for i in range(2, min(len(seq), max_len + 1)):
                full_sequences.append((user2idx[uid], tuple(seq[:i])))

    print(f"完整训练样本数: {len(full_sequences)}")

    full_dataset = SASRecDataset(full_sequences, num_items, max_len, 10)
    full_loader = DataLoader(full_dataset, batch_size=128, shuffle=True, num_workers=0)

    if os.path.exists('best_model_multimodal.pth'):
        model.load_state_dict(torch.load('best_model_multimodal.pth'))
        print("已加载最佳模型")

    # 微调
    fine_tune_epochs = 5
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(fine_tune_epochs):
        loss = train_epoch(model, full_loader, optimizer, item_features, device)
        print(f"Fine-tune Epoch {epoch + 1}/{fine_tune_epochs} | Loss: {loss:.4f}")

    # 计算热门商品
    item_popularity = defaultdict(int)
    for seq in full_user_seq.values():
        for item in seq:
            item_popularity[item] += 1
    popular_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    popular_item_ids = [idx2item[idx] for idx, _ in popular_items[:50]]

    # 生成推荐
    model.eval()
    submission = []

    with torch.no_grad():
        for uid in tqdm(test_users_df['user_id'], desc="Generating Recommendations"):
            if uid not in full_user_seq or len(full_user_seq[uid]) == 0:
                preds = popular_item_ids[:20]
            else:
                seq = full_user_seq[uid]
                if len(seq) <= max_len:
                    input_seq = [0] * (max_len - len(seq)) + seq
                else:
                    input_seq = seq[-max_len:]

                seq_tensor = torch.LongTensor([input_seq]).to(device)
                scores = model.predict_all(seq_tensor, item_features).squeeze(0).cpu().numpy()

                if len(scores) != num_items:
                    full_scores = np.zeros(num_items)
                    min_len = min(len(scores), num_items)
                    full_scores[:min_len] = scores[:min_len]
                    scores = full_scores

                seq_set = set(seq)
                for idx in seq_set:
                    if idx < num_items:
                        scores[idx] = -np.inf

                top_indices = np.argsort(scores)[::-1][:20]
                preds = [idx2item[idx] for idx in top_indices if idx < num_items and not np.isinf(scores[idx])]
                if len(preds) < 20:
                    for item in popular_item_ids:
                        if item not in preds:
                            preds.append(item)
                            if len(preds) == 20:
                                break
            submission.append(' '.join(map(str, preds[:20])))
    submission_df = pd.DataFrame({
        'user_id': test_users_df['user_id'],
        'prediction': submission
    })
    submission_df.to_csv('submission_sasrec.csv', index=False)
    print(f"\n提交文件已保存: submission_sasrec.csv")
    print(f"最佳验证NDCG@20: {best_ndcg:.4f}")
    return best_ndcg


# ID模型 vs 多模态模型对比实验
def run_model_comparison():
    print("\n" + "=" * 80)
    print("ID模型 vs 多模态模型性能对比实验")
    print("=" * 80)
    results = {}
    # 训练多模态模型
    print("\n训练多模态模型...")
    results['multimodal'] = train_single_model(use_multimodal=True)
    # 训练ID模型
    print("\n训练ID模型...")
    results['id_only'] = train_single_model(use_multimodal=False)
    plot_comparison(results)
    comparison_df = pd.DataFrame({
        'Model': ['ID Only', 'Multimodal'],
        'Best_NDCG@20': [results['id_only']['best_ndcg'], results['multimodal']['best_ndcg']],
        'Final_NDCG@20': [results['id_only']['final_ndcg'], results['multimodal']['final_ndcg']],
        'Best_Epoch': [results['id_only']['best_epoch'], results['multimodal']['best_epoch']]
    })
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("\n对比结果已保存到: model_comparison_results.csv")
    return results


def train_single_model(use_multimodal=True):
    """训练单个模型（ID only 或 多模态）"""
    model_name = "Multimodal" if use_multimodal else "ID Only"

    # 超参数配置
    max_len = 50
    batch_size = 128
    embed_size = 256
    num_epochs = 10  #30,降低提高效率
    learning_rate = 0.001
    neg_samples = 10
    num_heads = 4
    num_layers = 2
    dropout = 0.2

    # 创建数据集
    train_dataset = SASRecDataset(train_sequences, num_items, max_len, neg_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建模型
    model = SASRecModel(
        num_items=num_items,
        item_features_dim=257,
        embed_size=embed_size,
        max_len=max_len,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_multimodal=use_multimodal
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # 训练循环
    best_ndcg = 0.0
    best_epoch = 0
    patience = 5
    wait = 0
    epoch_ndcgs = []

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, item_features, device)

        if (epoch + 1) % 3 == 0:
            val_ndcg = evaluate(model, valid_df, user_seq, item_features, max_len, K=20)
            epoch_ndcgs.append(val_ndcg)
            print(f"{model_name} | Epoch {epoch + 1:2d}/{num_epochs} | Loss: {loss:.4f} | NDCG@20: {val_ndcg:.4f}")

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f'best_model_{model_name.lower()}.pth')
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            print(f"{model_name} | Epoch {epoch + 1:2d}/{num_epochs} | Loss: {loss:.4f}")

        scheduler.step()

    print(f"{model_name} | 最佳验证NDCG@20: {best_ndcg:.4f} (Epoch {best_epoch})")

    return {
        'model_name': model_name,
        'best_ndcg': best_ndcg,
        'best_epoch': best_epoch,
        'final_ndcg': val_ndcg if 'val_ndcg' in locals() else best_ndcg,
        'epoch_ndcgs': epoch_ndcgs
    }


def plot_comparison(results):
    """绘制模型对比图"""
    plt.figure(figsize=(12, 5))

    # 子图1: NDCG曲线对比，对齐两组数据长度
    id_ndcg = results['id_only']['epoch_ndcgs']
    multi_ndcg = results['multimodal']['epoch_ndcgs']
    min_len = min(len(id_ndcg), len(multi_ndcg))
    id_ndcg = id_ndcg[:min_len]
    multi_ndcg = multi_ndcg[:min_len]
    epochs = [3 * (i + 1) for i in range(min_len)]

    plt.subplot(1, 2, 1)
    plt.plot(epochs, id_ndcg, 'o-', label='ID Only', linewidth=2, markersize=6)
    plt.plot(epochs, multi_ndcg, 's-', label='Multimodal', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('NDCG@20', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 子图2: 最佳指标柱状图对比
    plt.subplot(1, 2, 2)
    models = ['ID Only', 'Multimodal']
    best_scores = [results['id_only']['best_ndcg'], results['multimodal']['best_ndcg']]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(models, best_scores, color=colors, alpha=0.8)
    plt.ylabel('Best NDCG@20', fontsize=12)
    plt.title('Best Performance Comparison', fontsize=14)

    # 柱子标注数值
    for bar, score in zip(bars, best_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=11)

    plt.ylim(0, max(best_scores) * 1.15)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("对比图已保存到: model_comparison.png")


# 超参数影响分析
def run_hyperparameter_analysis():
    """超参数影响分析"""
    print("\n" + "=" * 80)
    print("超参数影响分析实验")
    print("=" * 80)

    # 定义要测试的超参数
    hyperparams_to_test = {
        'embed_size': [128, 256, 512],
        'dropout': [0.1, 0.2, 0.3, 0.5],
        'num_layers': [1, 2, 3],
        'num_heads': [2, 4, 8]
    }

    all_results = {}

    # 测试每个超参数
    for param_name, param_values in hyperparams_to_test.items():
        print(f"\n测试超参数: {param_name}")
        param_results = []

        for value in param_values:
            print(f"\n  测试 {param_name} = {value}")

            # 设置超参数
            max_len = 50
            batch_size = 128
            embed_size = 256 if param_name != 'embed_size' else value
            num_epochs = 15  # 减少轮数加快实验
            learning_rate = 0.001
            neg_samples = 10
            num_heads = 4 if param_name != 'num_heads' else value
            num_layers = 2 if param_name != 'num_layers' else value
            dropout = 0.2 if param_name != 'dropout' else value

            # 创建数据集
            train_dataset = SASRecDataset(train_sequences, num_items, max_len, neg_samples)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            # 创建模型
            model = SASRecModel(
                num_items=num_items,
                item_features_dim=257,
                embed_size=embed_size,
                max_len=max_len,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                use_multimodal=True
            ).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

            # 快速训练
            best_ndcg = 0.0
            patience = 3
            wait = 0

            for epoch in range(num_epochs):
                loss = train_epoch(model, train_loader, optimizer, item_features, device)

                if (epoch + 1) % 3 == 0:
                    val_ndcg = evaluate(model, valid_df, user_seq, item_features, max_len, K=20)
                    print(f"    Epoch {epoch + 1}/{num_epochs} | NDCG@20: {val_ndcg:.4f}")

                    if val_ndcg > best_ndcg:
                        best_ndcg = val_ndcg
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break
                else:
                    if (epoch + 1) % 3 == 0:
                        print(f"    Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.4f}")

                scheduler.step()

            param_results.append({
                'param_value': value,
                'best_ndcg': best_ndcg
            })
            print(f"    最佳NDCG@20: {best_ndcg:.4f}")

        all_results[param_name] = param_results

    # 绘制超参数影响图
    plot_hyperparameter_effects(all_results)

    # 保存结果到CSV
    save_hyperparameter_results(all_results)

    return all_results


def plot_hyperparameter_effects(all_results):
    """绘制超参数影响图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    param_titles = {
        'embed_size': 'Embedding Dimension',
        'dropout': 'Dropout Rate',
        'num_layers': 'Number of Transformer Layers',
        'num_heads': 'Number of Attention Heads'
    }

    for idx, (param_name, results) in enumerate(all_results.items()):
        ax = axes[idx]

        values = [r['param_value'] for r in results]
        scores = [r['best_ndcg'] for r in results]

        # 绘制折线图
        ax.plot(values, scores, 'o-', color='#4ECDC4', linewidth=2, markersize=8, markerfacecolor='#FF6B6B')

        # 标注最佳点
        best_idx = np.argmax(scores)
        ax.plot(values[best_idx], scores[best_idx], 'D', color='gold', markersize=12, zorder=5)
        ax.annotate(f'Best: {scores[best_idx]:.4f}',
                    xy=(values[best_idx], scores[best_idx]),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold')

        ax.set_xlabel(param_titles[param_name], fontsize=12)
        ax.set_ylabel('NDCG@20', fontsize=12)
        ax.set_title(f'Impact of {param_titles[param_name]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 如果是分类变量，设置x轴为整数
        if param_name in ['num_layers', 'num_heads', 'embed_size']:
            ax.set_xticks(values)

    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("超参数影响图已保存到: hyperparameter_analysis.png")


def save_hyperparameter_results(all_results):
    for param_name, results in all_results.items():
        df = pd.DataFrame(results)
        df.columns = [param_name, 'best_ndcg']
        df.to_csv(f'hyperparam_{param_name}_results.csv', index=False)
        print(f"超参数 {param_name} 结果已保存到: hyperparam_{param_name}_results.csv")
    summary_data = []
    for param_name, results in all_results.items():
        best_result = max(results, key=lambda x: x['best_ndcg'])
        summary_data.append({
            'Parameter': param_name,
            'Best_Value': best_result['param_value'],
            'Best_NDCG@20': best_result['best_ndcg']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('hyperparameter_summary.csv', index=False)
    print("\n超参数汇总结果已保存到: hyperparameter_summary.csv")


def main():
    print("=" * 80)
    print("智能推荐系统实验功能选择")
    print("=" * 80)
    print("1. 基础训练 + 生成测试提交文件")
    print("2. ID模型 vs 多模态模型性能对比实验")
    print("3. 超参数影响分析实验")
    print("4. 退出程序")
    print("=" * 80)
    while True:
        user_choice = input("\n请输入要执行的功能编号（1-4）：").strip()
        if not user_choice.isdigit():
            print("输入无效，请输入1-4之间的数字！")
            continue
        choice_num = int(user_choice)
        if choice_num < 1 or choice_num > 4:
            print("编号超出范围，请输入1-4之间的数字！")
            continue
        if choice_num == 1:
            print("\n开始执行：基础训练 + 生成提交文件")
            run_baseline_training()
            print("\n基础训练完成，提交文件已生成！")
        elif choice_num == 2:
            print("\n开始执行：ID模型 vs 多模态模型性能对比实验")
            run_model_comparison()
            print("\n模型对比实验完成，结果表格与图表已保存！")
        elif choice_num == 3:
            print("\n开始执行：超参数影响分析实验")
            run_hyperparameter_analysis()
            print("\n超参数分析实验完成，结果表格与图表已保存！")
        elif choice_num == 4:
            print("\n程序退出，感谢使用！")
            break
        continue_choice = input("\n是否继续执行其他功能？（y/n）：").strip().lower()
        if continue_choice != 'y':
            print("\n程序退出，感谢使用！")
            break
if __name__ == "__main__":
    main()