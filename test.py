import torch
import tqdm
from torch.utils import data
from TransE import TransE

class TestDataset(data.Dataset):
    def __init__(self, ent2id, rel2id, test_data_list):
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.data = test_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, relation = self.data[index]
        head_id = self.ent2id[head]
        relation_id = self.rel2id[relation]
        return head_id, relation_id

# 读取数据
with open('OpenBG500/OpenBG500_entity2text.tsv', 'r', encoding='utf-8') as fp:
    dat = fp.readlines()
    lines = [line.strip('\n').split('\t') for line in dat]
ent2id = {line[0]: i for i, line in enumerate(lines)}
id2ent = {i: line[0] for i, line in enumerate(lines)}

with open('OpenBG500/OpenBG500_relation2text.tsv', 'r', encoding='utf-8') as fp:
    dat = fp.readlines()
    lines = [line.strip().split('\t') for line in dat]
rel2id = {line[0]: i for i, line in enumerate(lines)}

with open('OpenBG500/OpenBG500_test.tsv', 'r', encoding='utf-8') as fp:
    test = fp.readlines()
    test = [line.strip('\n').split('\t') for line in test]

# 构建数据集
test_dataset = TestDataset(ent2id, rel2id, test)
test_data_loader = data.DataLoader(test_dataset, batch_size=20)

# 构建模型
model = TransE(len(ent2id), len(rel2id), norm=3, dim=100).cuda()
model.load_state_dict(torch.load('transE.pth'))
model.eval()

predict_all = []
for heads, relations in tqdm.tqdm(test_data_loader):
    # 预测的id,结果为tensor(batch_size*10)
    predict_id = model.link_predict(heads.cuda(), relations.cuda())
    # 结果取到cpu并转为一行的list以便迭代
    predict_list = predict_id.cpu().numpy().reshape(1,-1).squeeze(0).tolist()
    # id转为实体
    predict_ent = list(map(lambda x: id2ent[x], predict_list))
    # 保存结果
    predict_all.extend(predict_ent)

# 写入文件，按提交要求
with open('submission.tsv', 'w', encoding='utf-8') as f:
    for i in range(len(test)):
        # 直接writelines没有空格分隔，手工加分割符，得按提交格式来
        list = [x + '\t' for x in test[i]] + [x + '\n' if i == 9 else x + '\t' for i, x in enumerate(predict_all[i*10:i*10+10])]
        f.writelines(list)
print('file saved !')
