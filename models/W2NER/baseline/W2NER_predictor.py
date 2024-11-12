import argparse
import torch
import torch.autograd
import os

# import sys
# sys.path.append('LLM_Project/CMeKG_tools-main/W2NER/baseline')

# from .model import Model
# from .data_loader import *
# from .utils import *

from models.W2NER.baseline.model import Model
import models.W2NER.baseline.data_loader as data_loader
import models.W2NER.baseline.utils as utils
from models.W2NER.baseline import config
# from W2NER import *

class Trainer(object):
    def __init__(self, model, config, args):
        self.model = model
        self.config = config
        self.args = args

    def predict(self, epoch, dataset, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        i = 0
        with torch.no_grad():
            for data_batch in dataset:      # fixme
                sentence_batch = data[0]
                data_batch = data_loader.collate_fn([data_batch])
                entity_text = data_batch[-1]
                data_batch = [data.to(self.args.device) for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence_batch['sentence']
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": self.config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        return result[0]['entity']

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path), strict=False)
        # model.load_state_dict(state_dict, False)

def load_W2NER_model(device):
    # print(os.getcwd())
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default=r'/data1/wzy/test3/CMeEE/baseline/config/cmeee.json')
    # parser.add_argument('--save_path', type=str, default=r'/data1/wzy/test3/CMeEE/baseline/basemodel.pt')

    parser.add_argument('--config', type=str, default=r'./models/W2NER/baseline/config/cmeee.json')
    parser.add_argument('--save_path', type=str, default=r'./models/W2NER/baseline/basemodel.pt')

    parser.add_argument('--predict_path', type=str, default=os.getcwd()+r'/output.json')
    parser.add_argument('--device', type=int, default=1)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    from .config import Config

    config = Config(args)

    logger = utils.get_logger(config.dataset)
    # logger.info(config)
    config.logger = logger
    config.label_num = 11

    # args.device = str(device).split(":")[-1]
    args.device = device

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Building Model")

    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model, config, args)

    # print(model.state_dict().keys())  # 打印模型的键
    # print(model.named_parameters())  # 打印模型的参数名称

    trainer.load(config.save_path)

    return (trainer.model, config, args)

def predict(sentence, model, config, args):
    trainer = Trainer(model, config, args)
    dataset, ori_data = data_loader.load_data_bert(config, data=sentence)

    # test_loader = (
    #     DataLoader(dataset=dataset,
    #                batch_size=config.batch_size,
    #                collate_fn=data_loader.collate_fn,
    #                shuffle=False,
    #                num_workers=1,
    #                drop_last=False)
    # )

    entity_list = trainer.predict("Final", dataset, ori_data)

    out_list = [item['text'] for item in entity_list]

    merged_data = [''.join(data) for data in out_list]

    return merged_data

# model, config = load_W2NER_model()
#
# sentence = '以下是中国医师考试中规培结业考试的一道多项选择题，请分析每个选项，并最后给出答案。HIV可以感染的细胞有A. CD4+T细胞B. 巨噬细胞C. 树突状细胞D. B细胞接受数据： A. CD4+T细胞：HIV主要感染的是辅助性T细胞，尤其是CD4+T细胞。B. 巨噬细胞：巨噬细胞也可以被HIV感染，但并非其主要靶细胞。C. 树突状细胞：树突状细胞也可以被HIV感染，但同样非其主要靶细胞。D. B细胞：B细胞不被HIV感染。答案：A. CD4+T细胞'
# list = predict(sentence, model, config)
# print(list)