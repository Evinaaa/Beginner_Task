def train(model, optimizer, train_triples, n_epochs=100):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for h, r, t in DataLoader(train_triples, batch_size=1024):
            # 正样本得分
            pos_scores = model.score(h, r, t)
            
            # 生成负样本
            neg_samples = negative_sampling((h,r,t), entities)
            neg_scores = model.score(neg_samples)
            
            # TransE/RotatE使用Margin Loss
            loss = F.margin_ranking_loss(
                pos_scores, neg_scores, 
                target=torch.ones_like(pos_scores),
                margin=1.0
            )
            
            # ConvE使用BCEWithLogitsLoss
            # loss = F.binary_cross_entropy_with_logits(...)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss {total_loss}")

def evaluate(model, test_triples):
    ranks = []
    for h, r, t in test_triples:
        # 破坏尾实体
        all_t = model.ent_emb.weight
        scores = model.score(h.expand_as(all_t), r.expand_as(all_t), all_t)
        rank = (scores > model.score(h, r, t)).sum() + 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks)
    mrr = (1.0 / ranks).mean()
    hits10 = (ranks <= 10).float().mean()
    return {"MRR": mrr, "Hits@10": hits10}