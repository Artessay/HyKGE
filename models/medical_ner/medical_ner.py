# coding:utf-8
import codecs
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import json
from .utils import load_vocab
from .ner_constant import *
import torch.nn.functional as F
from .model_ner import BERT_LSTM_CRF
import os
import warnings

# 将所有警告设置为忽略模式
warnings.filterwarnings("ignore")


class medical_ner(object):
    def __init__(self, use_cuda, device):
        self.NEWPATH = './models/medical_ner/model.pkl'
        self.vocab = load_vocab('./models/medical_ner/vocab.txt')
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.model = BERT_LSTM_CRF('./models/medical_ner', tagset_size, 768, 200, 2,
                              dropout_ratio=0.5, dropout1=0.5, use_cuda=use_cuda)
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.to(device)

    def from_input(self, input_str):
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str] + ['[SEP]']
        raw_text.append(text)
        cur_len = len(text)
        cur_len = len([self.vocab[x] for x in text if self.vocab.__contains__(x)])
        # raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len)
        raw_textid = [self.vocab[x] for x in text if self.vocab.__contains__(x)] + [0] * (max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (max_length - cur_len)
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def from_txt(self, input_path):
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip())==0:
                    continue
                if len(line) > 448:
                    line = line[:448]
                temptext = ['[CLS]'] + [x for x in line[:-1]] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext[:cur_len]] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength
    def split_entity_input(self,label_seq):
        entity_mark = dict()
        entity_pointer = None
        for index, label in enumerate(label_seq):
            #print(f"before: {label_seq}")
            if label.split('-')[-1]=='B':
                category = label.split('-')[0]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
            elif label.split('-')[-1]=='M':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            elif label.split('-')[-1]=='E':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            else:
                entity_pointer = None
           # print(entity_mark)
        return entity_mark
    def predict_sentence(self, sentence):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        if sentence == '':
            print("输入为空！请重新输入")
            return []
        if len(sentence) > 448:
            print("输入句子过长，请输入小于148的长度字符！")
            sentence = sentence[:448]
        sentence = sentence.lower()
        raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device), strict=False)
        self.model.eval()
        self.model.to(device)

        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)

            if self.use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)

            predict_tags = self.model(sentence, masks)      # Note
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]
            entity_mark = self.split_entity_input(pred)
            # entity_list = {}
            entity_list = []
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    # print(item, ent)
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)

                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    # entity_list[tag_dic[tag]] = entity
                    entity_list.append(entity)

            # print(entity_list)
        return entity_list

    def predict_file(self, input_file, output_file):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device))
        self.model.eval()
        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if self.use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]

            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)
                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            op_file.write("".join(raw_text))
            op_file.write("\n")
            op_file.write(json.dumps(entity_list, ensure_ascii=False))
            op_file.write("\n")

        op_file.close()
        print('处理完成！')
        print("结果保存至 {}".format(output_file))


if __name__ == "__main__":
    my_pred = medical_ner(use_cuda=True, device='cuda:0')
    # sentence = "抑郁症受遗传的影响。在抑郁症青少年中，约25%～33%的家庭有一级亲属的发病史，是没有抑郁症青少年家庭发病的2倍。"
    # sentence = "我感觉到抑郁了，你知道怎么治疗吗？需要吃青霉素吗。"
    # sentence = "孕中期腹泻见红，吃药好转您好，我怀孕六个月，前天不知道怎么的，肚子痛，拉肚子，而且还阴道见红了"
    #
    # res = my_pred.predict_sentence(sentence)

    my_pred.predict_sentence("2-氯脱氧腺苷")
    my_pred.predict_sentence("5-羟吲哚乙酸")
    a = my_pred.model.lstm_feats
    print(a)

    # print("---")
    # print(res)


    # my_pred.predict_file("my_test.txt", "outt.txt")