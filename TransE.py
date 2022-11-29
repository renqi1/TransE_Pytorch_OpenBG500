import torch
from torch import nn
from torch.utils import data
import numpy as np
import tqdm

class TripleDataset(data.Dataset):

    def __init__(self, ent2id, rel2id, triple_data_list):
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.data = triple_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, relation, tail = self.data[index]
        head_id = self.ent2id[head]
        relation_id = self.rel2id[relation]
        tail_id = self.ent2id[tail]
        return head_id, relation_id, tail_id

class TransE(nn.Module):

    def __init__(self, entity_num, relation_num, norm=1, dim=100):
        super(TransE, self).__init__()
        self.norm = norm
        self.dim = dim
        self.entity_num = entity_num
        self.entities_emb = self._init_emb(entity_num)
        self.relations_emb = self._init_emb(relation_num)

    def _init_emb(self, num_embeddings):
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        embedding.weight.data = torch.div(embedding.weight.data, embedding.weight.data.norm(p=2, dim=1, keepdim=True))
        return embedding

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        return positive_distances, negative_distances

    def _distance(self, triplets):
        heads = self.entities_emb(triplets[:, 0])
        relations = self.relations_emb(triplets[:, 1])
        tails = self.entities_emb(triplets[:, 2])
        return (heads + relations - tails).norm(p=self.norm, dim=1)

    def link_predict(self, head, relation, tail=None, k=10):
        # h_add_r: [batch size, embed size] -> [batch size, 1, embed size] -> [batch size, entity num, embed size]
        h_add_r = self.entities_emb(head) + self.relations_emb(relation)
        h_add_r = torch.unsqueeze(h_add_r, dim=1)
        h_add_r = h_add_r.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # embed_tail: [batch size, embed size] -> [batch size, entity num, embed size]
        embed_tail = self.entities_emb.weight.data.expand(h_add_r.shape[0], self.entity_num, self.dim)
        # values: [batch size, k] scores, the smaller, the better
        # indices: [batch size, k] indices of entities ranked by scores
        values, indices = torch.topk(torch.norm(h_add_r - embed_tail, dim=2), k=self.entity_num, dim=1, largest=False)
        if tail is not None:
            tail = tail.view(-1, 1)
            rank_num = torch.eq(indices, tail).nonzero().permute(1, 0)[1]+1
            rank_num[rank_num > 9] = 10000
            mrr = torch.sum(1/rank_num)
            hits_1_num = torch.sum(torch.eq(indices[:, :1], tail)).item()
            hits_3_num = torch.sum(torch.eq(indices[:, :3], tail)).item()
            hits_10_num = torch.sum(torch.eq(indices[:, :10], tail)).item()
            return mrr, hits_1_num, hits_3_num, hits_10_num     # 返回一个batchsize, mrr的和，hit@k的和
        return indices[:, :k]
    
    def evaluate(self, data_loader, dev_num=5000):
        mrr_sum = hits_1_nums = hits_3_nums = hits_10_nums = 0
        for heads, relations, tails in tqdm.tqdm(data_loader):
            mrr_sum_batch, hits_1_num, hits_3_num, hits_10_num = self.link_predict(heads.cuda(), relations.cuda(), tails.cuda())
            mrr_sum += mrr_sum_batch
            hits_1_nums += hits_1_num
            hits_3_nums += hits_3_num
            hits_10_nums += hits_10_num
        return mrr_sum/dev_num, hits_1_nums/dev_num, hits_3_nums/dev_num, hits_10_nums/dev_num


if __name__ == '__main__':
    # 读取数据
    with open('OpenBG500/OpenBG500_entity2text.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        lines = [line.strip('\n').split('\t') for line in dat]
    ent2id = {line[0]: i for i, line in enumerate(lines)}
    with open('OpenBG500/OpenBG500_relation2text.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        lines = [line.strip().split('\t') for line in dat]
    rel2id = {line[0]: i for i, line in enumerate(lines)}
    with open('OpenBG500/OpenBG500_train.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        train = [line.strip('\n').split('\t') for line in dat]
    with open('OpenBG500/OpenBG500_dev.tsv', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        dev = [line.strip('\n').split('\t') for line in dat]

    # 参数设置
    train_batch_size = 100000  # batchsize增大，得分略有上升
    dev_batch_size = 20  # 显存不够就调小
    epochs = 40
    margin = 1
    print_frequency = 5  # 每多少step输出一次信息
    validation = True  # 是否验证，验证比较费时，注意loss不是越小效果越好哦!!!
    dev_interval = 5  # 每多少轮验证一次
    best_mrr = 0
    learning_rate = 0.001  # 学习率建议粗调0.01-0.001，精调0.001-0.0001
    distance_norm = 3  # 论文是L1距离效果不好，取2或3效果好
    embedding_dim = 100  # 维度增大可能会有提升

    # 构建数据集
    train_dataset = TripleDataset(ent2id, rel2id, train)
    dev_dataset = TripleDataset(ent2id, rel2id, dev)
    train_data_loader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_data_loader = data.DataLoader(dev_dataset, batch_size=dev_batch_size)

    # 构建模型
    model = TransE(len(ent2id), len(rel2id), norm=distance_norm, dim=embedding_dim).cuda()
    # model.load_state_dict(torch.load('transE.pth'))

    # mrr, hits1, hits3, hits10 = model.evaluate(dev_data_loader)
    # print(f'mrr: {mrr}, hit@1: {hits1}, hit@3: {hits3}, hit@10: {hits10}')

    # 优化器adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 损失函数， 对于本例，loss=max(0, (pd-nd)+1)， 负样本距离越小，正样本距离越大越好
    criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')

    print(f"start train")
    model.train()
    for epoch in range(epochs):
        all_loss = 0
        for i, (local_heads, local_relations, local_tails) in enumerate(train_data_loader):

            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1).cuda()

            # 生成负样本
            head_or_tail = torch.randint(high=2, size=local_heads.size())
            random_entities = torch.randint(high=len(ent2id), size=local_heads.size())
            broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1).cuda()

            # # 生成负样本, 只打乱tail
            # random_entities = torch.randint(high=len(ent2id), size=local_heads.size())
            # negative_triples = torch.stack((random_entities, local_relations, random_entities), dim=1).cuda()

            optimizer.zero_grad()
            pd, nd = model(positive_triples, negative_triples)
            # pd要尽可能小， nd要尽可能大
            loss = criterion(pd, nd, torch.tensor([-1], dtype=torch.long).cuda())
            loss.backward()
            all_loss += loss.data
            optimizer.step()
            if i % print_frequency == 0:
                print(f"epoch:{epoch}/{epochs}, step:{i}/{len(train_data_loader)}, loss={loss.item()}, avg_loss={all_loss / (i + 1)}")
        print(f"epoch:{epoch}/{epochs}, all_loss={all_loss}")

        # 验证
        if validation and (epoch+1) % dev_interval == 0:
            print('testing...')
            improve = ''
            mrr, hits1, hits3, hits10 = model.evaluate(dev_data_loader)
            if mrr >= best_mrr:
                best_mrr = mrr
                improve = '*'
                torch.save(model.state_dict(), 'transE_best.pth')
            torch.save(model.state_dict(), 'transE_latest.pth')
            print(f'mrr: {mrr}, hit@1: {hits1}, hit@3: {hits3}, hit@10: {hits10}  {improve}')
        if not validation:
            torch.save(model.state_dict(), 'transE_latest.pth')
