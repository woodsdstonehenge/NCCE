#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
from tqdm import tqdm
import torch
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_bert import BertConfig
import torch.nn as nn
from sklearn.metrics import f1_score
import pickle as pkl

from utils import MyDataLoader
from model import myLSTM
from metrics_back import CorefEvaluator
from utils import get_f1_by_bio_nomask
from utils import get_mention_f1

try:
    from apex import amp
except:
    pass


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import random
import numpy as np

random.seed(42)
torch.random.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)


class Pipeline:
    def __init__(self, args):
        self.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.args = args
        basename = os.path.basename(os.getcwd())
        basename = 'scripts'
        config = json.load(open('config.json', 'r'))
        for cfg in config:
            self.__setattr__(cfg, config[cfg])
        self.data_dir = self.data_dir.format(basename)
        self.target_dir = self.target_dir.format(basename)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def execute_iter(self, training=True, test=False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        # path = os.path.join(self.target_dir, 'best_20.pth.tar')
        # self.model.load_state_dict(torch.load(path)['model'])
        dataloader = self.validLoader
        if training:
            dataloader = self.trainLoader
        elif test:
            dataloader = self.testLoader

        dataiter = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)
        bio_predict, bio_gold, mention_predict, mention_gold = [], [], [], []
        sx, f1s, losses, fusion_matrix = [], [], [], []
        score_1 = []
        coref_evaluator = CorefEvaluator()
        detail_prf = []
        rate_res = []
        cluster_res = []

        for index, data in enumerate(dataiter):
            self.model.valid_index = index
            #print('mention', data['mention_sets'])
            input_ids = data['input_ids'][0].to(self.device)
            input_masks = data['input_masks'][0].to(self.device)
            input_segments = data['input_segments'][0].to(self.device)
            input_labels = data['input_labels'][0].to(self.device)
            mention_sets = data['mention_sets']
            input_poses = data['input_poses'][0].to(self.device)
            input_sememes = data['input_sememes'].to(self.device)
            input_sememes_nums = data['input_sememes_nums'][0].to(self.device)
            # use_pos = self.args.use_pos
            use_pos = True
            if training:
                loss, correct_count, count, bio_p, bio_g, mention_p, mention_g = self.model(input_ids, input_masks,
                                                                                            input_segments,
                                                                                            input_labels, mention_sets,
                                                                                            input_poses, input_sememes,
                                                                                            input_sememes_nums, use_pos,
                                                                                            False, coref_evaluator)
                if self.fp16 == 'use':
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    nn.utils.clip_grad_norm(amp.master_params(self.optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad_norm)
                self.scheduler.step()
                self.optimizer.step()
                self.model.zero_grad()
            else:
                with torch.no_grad():
                    loss, correct_count, count, bio_p, bio_g, mention_p, mention_g = \
                        self.model(input_ids, input_masks, input_segments, input_labels, mention_sets, input_poses,
                                   input_sememes, input_sememes_nums, use_pos=use_pos, show_res=True,
                                   coref_evaluator=coref_evaluator)
                    rate_res.append(self.model.rate)
            bio_predict.append(bio_p)
            bio_gold.append(bio_g)
            # p1, r1, f1 = get_f1_by_bio_nomask(bio_p, bio_g)

            # p2, r2, f2 = get_mention_f1(mention_p, mention_g)

            f1, fusion = get_f1_by_bio_nomask(bio_p, bio_g)
            score_1.append(f1)
            fusion_matrix.append(fusion)
            f1s.append(f1)
            losses.append(loss.item())
            prf = coref_evaluator.get_prf()
            sub_score = coref_evaluator.get_sub_score()

            description = "Epoch {},loss:{:.3f}, label bio f1:{:.3f}".format(self.global_epcoh, np.mean(losses),
                                                                             np.mean(f1s, 0)[-1])
            description = "Epoch {},loss:{:.3f}, label bio f1:{:.3f}, mean: p {:.4f}, r {:.4f}, f {:.4f}".format(
                self.global_epcoh, np.mean(losses), np.mean(score_1, 0)[-1], *prf)

            dataiter.set_description(description)
            detail_prf.append(self.model.single_prf)
            cluster_res.append(self.model.cluster)
        fusion_matrix = np.sum(fusion_matrix, 0)
        p = fusion_matrix[0] / fusion_matrix[1] if fusion_matrix[1] > 0 else 0
        r = fusion_matrix[0] / fusion_matrix[2] if fusion_matrix[2] > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        print(f)
        res = [(p, r, f), sub_score[0], sub_score[1], sub_score[2], prf]
        coref_evaluator.get_res()
        #pkl.dump(detail_prf, open('detail_prf.pkl', 'wb'))
        #pkl.dump(detail_prf, open('detail_prf.pkl', 'wb'))
        #pkl.dump(rate_res, open('rate_res.pkl', 'wb'))
        #pkl.dump(cluster_res, open('bert_hownet_gcn_joint.pkl', 'wb'))
        # exit(0)

        return res

    def forward(self):
        best_score, best_iter = 0, 0
        res = []
        for epoch in range(self.epoch_size):
            self.global_epcoh = epoch
            self.execute_iter()
            res_valid = self.execute_iter(training=False)
            res_test = self.execute_iter(training=False, test=True)


            res.append((res_valid, res_test))

            tmp = sorted(res, key=lambda x: x[0][-1][-1])[::-1]
            best_valid = tmp[0]
            score = best_valid[0][-1][-1]
            best_valid = [w for line in best_valid[1] for w in line]
            print("Best valid")
            print('item  bio    B3    CEAF  Mean')
            print('mat     ' + ('p     r     f     ' * 6))
            t = ([w for line in tmp[0][0] for w in line], [w for line in tmp[0][1] for w in line])
            print("valid " + ("{:.2f}," * 15).format(*[w * 100 for w in t[0]]))
            print("test  " + ("{:.2f}," * 15).format(*[w * 100 for w in t[1]]))
            tmp = sorted(res, key=lambda x: x[1][-1][-1])[::-1]
            t = [w for line in tmp[0][1] for w in line]
            print("test  " + ("{:.2f}," * 15).format(*[w * 100 for w in t]))

            self.best_score_nine = best_valid
            print("this score", score)

            if score > best_score:
                best_score = score
                best_iter = epoch
                torch.save({'epoch': epoch,
                            'model': self.model.cpu().state_dict(),
                            'best_score': best_score},
                           os.path.join(self.target_dir, "best_{}.pth.tar".format(epoch)))
                self.model.to(self.device)
                print("best score: ")
                print("score: {:.4f}".format(score))

            elif epoch - best_iter > self.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.patience))
                break
            # print("score: p: {:.4f}, r: {:.4f}, {:.4f}".format(p, r, score))

    def main(self):
        self.trainLoader = MyDataLoader(self, mode='train').getdata()
        self.validLoader = MyDataLoader(self, mode='valid').getdata()
        self.testLoader = MyDataLoader(self, mode='test').getdata()
        #self.testLoader = MyDataLoader(self, mode='final').getdata()
        
        #word_dict = pkl.load(open(os.path.join(self.data_dir, 'word_dict.pkl'), 'rb'))
        sememe_dict = pkl.load(open(os.path.join(self.data_dir, 'sememe_dict.pkl'), 'rb'))
        #embedding_matrix = pkl.load(open(os.path.join(self.data_dir, 'emb.pkl'), 'rb'))
        pos_dict = pkl.load(open(os.path.join(self.data_dir, 'pos_dict.pkl'), 'rb'))
        config = BertConfig.from_pretrained(self.bert_path, num_labels=3)
        config.__setattr__('bert_path', self.bert_path)
        config.__setattr__('num_pos', len(pos_dict))
        config.__setattr__('pos_emb_size', self.pos_emb_size)
        config.__setattr__('num_sememe', len(sememe_dict))
        config.__setattr__('sememe_emb_size', self.sememe_emb_size)
        self.model = myLSTM.from_pretrained(self.bert_path, config=config, device=self.device).to(self.device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0}
        ]
        self.optimizer = AdamW(self.model.parameters(),
                        lr=self.learning_rate,
                        eps=self.adam_epsilon, weight_decay=1e-6)
        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=self.warmup_steps,
                                        t_total=self.epoch_size * self.trainLoader.__len__())


        if self.fp16 == 'use':
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)
        self.criterion = nn.CrossEntropyLoss()
        self.forward()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=1)
    args = parser.parse_args()
    pipeline = Pipeline(args)
    pipeline.main()
