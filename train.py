ratings = ratings[ratings["rating"] >= 3.5]
ratings = ratings.sort_values(["userId", "timestamp"])

# Encode users & items
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ratings["uid"] = user_encoder.fit_transform(ratings["userId"])
ratings["iid"] = item_encoder.fit_transform(ratings["movieId"])

num_users = ratings["uid"].nunique()
num_items = ratings["iid"].nunique()

print("Users:", num_users)
print("Items:", num_items)
print("Interactions:", len(ratings))


# ================================
# 2. USER → POSITIVE ITEMS MAP
# ================================
user_pos_items = ratings.groupby("uid")["iid"].apply(set).to_dict()


# ================================
# 3. BPR SAMPLER
# ================================
def sample_bpr_batch(batch_size):
    users, pos_items, neg_items = [], [], []

    for _ in range(batch_size):
        u = random.choice(list(user_pos_items.keys()))
        pos = random.choice(list(user_pos_items[u]))

        while True:
            neg = random.randint(0, num_items - 1)
            if neg not in user_pos_items[u]:
                break

        users.append(u)
        pos_items.append(pos)
        neg_items.append(neg)

    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.LongTensor(neg_items)
    )


# ================================
# 4. LIGHTGCN (MF VERSION)
# ================================
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def score(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(dim=1)


# ================================
# 5. BPR LOSS (WITH REGULARIZATION)
# ================================
def bpr_loss(model, users, pos_items, neg_items, reg=1e-4):
    pos_scores = model.score(users, pos_items)
    neg_scores = model.score(users, neg_items)

    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    reg_loss = (
        model.user_emb(users).norm(2).pow(2) +
        model.item_emb(pos_items).norm(2).pow(2) +
        model.item_emb(neg_items).norm(2).pow(2)
    ) / users.size(0)

    return loss + reg * reg_loss


# ================================
# 6. TRAINING LOOP
# ================================
model = LightGCN(num_users, num_items, emb_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

EPOCHS = 20
STEPS_PER_EPOCH = 200
BATCH_SIZE = 1024

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for step in range(STEPS_PER_EPOCH):
        users, pos_items, neg_items = sample_bpr_batch(BATCH_SIZE)

        optimizer.zero_grad()
        loss = bpr_loss(model, users, pos_items, neg_items)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / STEPS_PER_EPOCH
    print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f}")


# ================================
# 7. SAVE MODEL & EMBEDDINGS
# ================================
torch.save(model.state_dict(), "lightgcn_mf.pt")

user_embeddings = model.user_emb.weight.detach().cpu().numpy()
item_embeddings = model.item_emb.weight.detach().cpu().numpy()

np.save("user_embeddings.npy", user_embeddings)
np.save("item_embeddings.npy", item_embeddings)

print("Training complete. Embeddings saved.")
