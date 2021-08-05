#!/usr/bin/env python

import OpenHowNet
import os
import pickle as pkl
import json
from collections import Counter
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from scipy import sparse

from pytorch_transformers.tokenization_bert import BertTokenizer
from stanfordcorenlp import StanfordCoreNLP
import logging
import random

class StanfordNLP:
    def __init__(self, host='http://172.16.133.173', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=15000, quiet=False, logging_level=logging.ERROR, lang='zh')   # , lang='zh' , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

class Preprocessor:

    def __init__(self, args, config_name=None):
        self.args = args

        basename = os.path.basename(os.getcwd())
        self.hownet = OpenHowNet.HowNetDict()
        #config_name = basename if config_name == None else config_name
        config = json.load(open('config.json', 'r', encoding='utf-8'))
        for cfg in config:
            self.__setattr__(cfg, config[cfg])
        #self.max_length = 512

        self.data_dir = self.data_dir.format(basename)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        #self.target_dir = self.target_dir.format(basename)
        self.stanfordnlp = StanfordNLP()

    def omit_char(self, content):
        """
        [
        {
        'text': '\ufeff2019年西安数字经济产业博览会盛大开幕',
        'event_chain': [
        {
        'event': [{'entity: {'start': [0], 'end':[16], 'text': '\ufeff2019年西安数字经济产业博览会盛大开幕'},
                 ''trigger':{'start': [0], 'end':[16], 'text': '\ufeff2019年西安数字经济产业博览会盛大开幕'}}]
        'chain_index
        }
        ]
        }]
        :param content:
        :return:
        """
        omits = ['\ufeff', ' ', '\u3000', '\xa0', '\u200b', '\u200c']
        res = []
        for sentence_index, sentence in enumerate(content):
            text = sentence['text']
            event_chain = sentence['event_chain']
            copy_chain = deepcopy(event_chain)

            for i in range(len(copy_chain)):
                for j in range(len(copy_chain[i]['event'])):
                    for k in range(len(copy_chain[i]['event'][j]['trigger']['start'])):
                        copy_chain[i]['event'][j]['trigger']['start'][k] = 0
                        copy_chain[i]['event'][j]['trigger']['end'][k] = 0
                    for k in range(len(copy_chain[i]['event'][j]['entity']['start'])):
                        copy_chain[i]['event'][j]['entity']['start'][k] = 0
                        copy_chain[i]['event'][j]['entity']['end'][k] = 0

            new_event_chain = []
            for index, word in enumerate(text):
                if word in omits:
                    for i in range(len(event_chain)):
                        for j in range(len(event_chain[i]['event'])):
                            for k in range(len(event_chain[i]['event'][j]['trigger']['start'])):
                                tmp = event_chain[i]['event'][j]['trigger']['start'][k]
                                if tmp > index:
                                    copy_chain[i]['event'][j]['trigger']['start'][k] += 1
                                tmp = event_chain[i]['event'][j]['trigger']['end'][k]
                                if tmp >= index:
                                    copy_chain[i]['event'][j]['trigger']['end'][k]  += 1

                            for k in range(len(event_chain[i]['event'][j]['entity']['start'])):
                                tmp = event_chain[i]['event'][j]['entity']['start'][k]
                                if tmp > index:
                                    copy_chain[i]['event'][j]['entity']['start'][k] += 1
                                tmp = event_chain[i]['event'][j]['entity']['end'][k]
                                if tmp >= index:
                                    copy_chain[i]['event'][j]['entity']['end'][k]  += 1

            for i in range(len(copy_chain)):
                for j in range(len(copy_chain[i]['event'])):
                    for k in range(len(copy_chain[i]['event'][j]['trigger']['start'])):
                        event_chain[i]['event'][j]['trigger']['start'][k] -= copy_chain[i]['event'][j]['trigger']['start'][k]
                        event_chain[i]['event'][j]['trigger']['end'][k] -= copy_chain[i]['event'][j]['trigger']['end'][k]
                    for k in range(len(copy_chain[i]['event'][j]['entity']['start'])):
                        event_chain[i]['event'][j]['entity']['start'][k] -= copy_chain[i]['event'][j]['entity']['start'][k]
                        event_chain[i]['event'][j]['entity']['end'][k] -= copy_chain[i]['event'][j]['entity']['end'][k]
            text = ''.join([w for w in text if not w in omits])
            res.append({'text': text, 'event_chain':event_chain})
        return res

    def have_labeled(self, sentence, start, end):
        '''
        if sentence[start] == 'B' and all([sentence[i] == 'I' for i in range(start + 1, end + 1)]):
            return False
        '''
        if all([sentence[i] == 'O' for i in range(start, end + 1)] ):
            return False
        return True

    def mytokenize(self, line):
        line = line.replace('\t', '。')
        location_dict = {}
        tokens = self.tokenizer.tokenize(line)
        i, j = 0, 0
        while i < len(tokens) or j < len(line):
            if tokens[i] == line[j].lower() or tokens[i] == self.UNK:
                location_dict[j] = i
                i += 1
                j += 1
            else:
                tmp_length = len(tokens[i].replace('#', ''))
                #print(line[j:j+tmp_length], tokens[i])
                #print("line", ord(line[j]))
                assert line[j:j+tmp_length].lower() == tokens[i].replace('#', '')
                for k in range(tmp_length):
                    location_dict[j+k] = i
                j += tmp_length
                i += 1
        return tokens, location_dict

    def execute(self, mode):
        path = os.path.join(self.source_dir, '{}.json'.format(mode))
        a = open(path, 'r', encoding='utf-8').read().splitlines()
        a = [json.loads(w) for w in a]
        if self.args.test_mode == 'test':
            a = a[:10]
        #a = a[:20]

        documents, mention_set, labels, part_of_speeches = [], [], [], []
        for document_index, sentences in tqdm(enumerate(a), total=len(a)):
            document, label, part_of_speech = [], [], []
            mention_dict = {}
            sentences = self.omit_char(sentences)
            # 遍历每一句话
            document_length = 0
            for sentence in sentences:
                text = sentence['text'][:self.max_length - 2]

                tokenized_text, location_dict = self.mytokenize(text)
                if len(tokenized_text) == 0 or (len(tokenized_text) == 1 and tokenized_text[0] == '。'):
                    continue
                sentence_pos = self.get_pos(tokenized_text)
                event_chain = sentence['event_chain']
                chain_index = [w['chain_index'] for w in event_chain]
                event_chain = [w['event'] for w in event_chain]
                bios = ['O' for w in tokenized_text]
                # 遍历一句话中的每一个事件链
                for sub_chain_index, sub_event in zip(chain_index, event_chain):
                    # 遍历一个链中的每一个事件
                    if sub_chain_index not in mention_dict:
                        mention_dict[sub_chain_index] = []
                    for event in sub_event:
                        entity = event['entity']
                        if len(entity['start']) == 0:
                            continue
                        start = entity['start'][-1]
                        end = entity['end'][-1]
                        if start > self.max_length - 3:
                            continue
                        if end > self.max_length - 3:
                            end = self.max_length - 3

                        start = location_dict[start]
                        end = location_dict[end]

                        if self.have_labeled(bios, start, end):
                            continue
                        bios[start] = 'B'
                        for i in range(start + 1, end+1):
                            bios[i] = 'I'

                        #start = location_dict[entity['start'][-1]] + document_length
                        #end = location_dict[entity['end'][-1]] + document_index
                        start = start + document_length
                        end = end + document_length
                        if (start, end) in mention_dict[sub_chain_index]:
                            continue
                        mention_dict[sub_chain_index].append((start, end))
                document_length += len(tokenized_text)
                label.append(bios)
                document.append(tokenized_text)
                part_of_speech.append(sentence_pos)
            documents.append(document)
            labels.append(label)
            mention_set.append([v for k, v in mention_dict.items()])
            part_of_speeches.append(part_of_speech)

        '''
        count = -1
        doc = documents[count]
        mention = mention_set[count]
        doc = [w for line in doc for w in line]
        #doc = ''.join(doc)
        for ment in mention:
            for mt in ment:
                print(doc[mt[0]:mt[1]+1])
                print(mt[0], mt[1])
            print('==')
        print('----')
        '''
        max_num = 0
        for mention in mention_set:
            for s in mention:
                max_num = max(max_num, len(s))
        print(max_num)

        return documents, labels, mention_set, part_of_speeches
    
    def build_dict(self, train_data, test_data, valid_data):
        
        train_document = train_data[0] + valid_data[0]
        test_document = test_data[0]

        word_count = {self.PAD : 1e10, self.UNK: 1e10-1}
        for sentences in train_document:
            for sentence in sentences:
                for word in sentence:
                    word_count[word] = word_count[word] + 1 if word in word_count else 1

        for sentences in test_document:
            for sentence in sentences:
                for word in sentence:
                    if word not in word_count:
                        word_count[word] = 0
                        
        word_count = Counter(word_count).most_common()
        word_dict = {w:i for i, (w, z) in enumerate(word_count)}
        path = os.path.join(self.data_dir, 'word_dict.pkl')
        pkl.dump(word_dict, open(path, 'wb'))
        self.word_dict = word_dict
    
    def build_embedding(self, word_dict, emb_file):
        embedding = {}
        doc = open(emb_file, 'r')
        for line in doc:
            line = line.split()
            word = line[0]
            if word in word_dict:
                embedding[word] = line[1:]
        vocab_size = len(word_dict)
        embedding_dim = len(list(embedding.values())[0])
        embedding_matrix = np.zeros([vocab_size, embedding_dim], dtype=np.float)
        for i, word in enumerate(word_dict):
            try:
                embedding_matrix[i] = embedding[word]
            except:
                embedding_matrix[i] = np.random.normal(embedding_dim)

        emb_path = os.path.join(self.data_dir, 'emb.pkl')
        pkl.dump(embedding_matrix, open(emb_path, 'wb'))

    def get_pos(self, line):
        return ['O' for w in line]
        sentence = ''.join(line)
        pos = self.stanfordnlp.ner(sentence)
        words = [w[0] for w in pos]
        words = ''.join(words)
        assert len(words) == len(sentence)
        #print(line)
        #print(pos)
        bio_label = []
        for po in pos:
            bio_label += ['I-' + po[1]] + ['I-'+po[1] for w in po[0][1:]]
            #bio_label += [po[1] for w in po[0]]
        assert len(words) == len(bio_label)
        res = []
        i, j = 0, 0
        while i < len(line):
            if line[i] == words[j]:
                res.append(bio_label[j])
            else:
                assert line[i] == words[j:j+len(line[i])]
                res.append(bio_label[j])
            j += len(line[i])
            i += 1
        for i, w in enumerate(line):
            if w == '[UNK]':
                res[i] = '[UNK]'
        assert len(res) == len(line)
        return res

    def build_label_dict(self, content):
        res = []
        res = [w for document in tqdm(content) for sentence in document for w in sentence if w != '[UNK]']
        res = Counter(res).most_common()[:10]
        res = {word:i + 4 for i, (word, freq) in enumerate(res)}
        res['[CLS]'] = 3
        res['[SEP]'] = 2
        res['[UNK]'] = 1
        res['[PAD]'] = 0
        path = os.path.join(self.data_dir, 'pos_dict.pkl')
        pkl.dump(res, open(path, 'wb'))
        self.pos_dict = res

    def get_sentence_index(self, sentences, max_length):
        indexs = []
        tmp_index = []
        tmp_count = 0
        for i, w in enumerate(sentences):
            if len(w) > max_length - 2:
                w = w[:max_length - 2]
            if tmp_count + len(w) <= max_length - 2:
                tmp_index.append(i)
                tmp_count += len(w)
            else:
                indexs.append(tmp_index)
                tmp_count = len(w)
                tmp_index = [i]
        if len(tmp_index) > 0:
            indexs.append(tmp_index)
        return indexs

    def get_sememe(self):
        sememes = self.hownet.get_all_sememes()
        self.sememes_dict = {w:i + 1 for i, w in enumerate(sememes)}
        self.sememes_dict['UNK'] = 0
        path = os.path.join(self.data_dir, 'sememe_dict.pkl')
        pkl.dump(self.sememes_dict, open(path, 'wb'))
    def get_first_sememe(self, word):
        sememe = self.hownet.get(word)
        res = []
        for semem in sememe:
            Def = semem['Def']
            Def = Def.strip('}').split(':')[0].split('|')[1]
            #Def = [self.sememes_dict[w if w in self.sememes_dict else 'UNK'] for w in Def]
            res.append(Def)
        res = [self.sememes_dict[w if w in self.sememes_dict else 'UNK'] for w in res]
        return res

    def get_sememe_and_num(self, sentence, max_num, first=True):
        sememes, nums = [], []
        sememes.append([])
        for word in sentence:
            if first:
                sememes.append(self.get_first_sememe(word))
                continue
            seme = self.hownet.get_sememes_by_word(word)
            tmp = []
            for sem in seme:
                tmp += [self.sememes_dict[w if w in self.sememes_dict else 'UNK'] for w in sem['sememes'][:1]]
            random.shuffle(tmp)
            sememes.append(tmp)
        sememes.append([])
        for i in range(self.max_length - len(sentence) - 2):
            sememes.append([])

        nums = list(map(len, sememes))
        if max_num == 0:
            max_num = max(nums)
        sememes = [w + [0 for w in range(max_num - len(w))] for w in sememes]
        sememes = [w[:max_num] for w in sememes]
        nums = [min(max_num, w) for w in nums]
        assert all(max_num == len(w) for w in sememes)
        return sememes, nums

    def get_sememe_adj(self, document):

        first_words = []
        all_sememes = []
        all_words = [w for line in document for w in line]
        for sentence in document:
            for word in sentence:
                sememe = list(set([w['Def'] for w in self.hownet.get(word)]))
                first_word = [w.split('}')[0].split(':')[0].split('|')[1].replace('}', '') for w in sememe]
                #print(word)
                #print(sememe)
                #[self.sememes_dict[w] for w in first_word]
                #print(first_word)
                all_sememe = list(set([tuple(w['sememes']) for w in self.hownet.get_sememes_by_word(word)]))
                #all_sememe = [w['sememes'] for w in all_sememe]
                all_sememe = [[z for z in p if z != w] for p, w in zip(all_sememe, first_word)]
                first_words.append(first_word)
                all_sememes.append(all_sememe)

        set_sememe = set([w for sent in all_sememes for line in sent for w in line] + [w for line in first_words for w in line])
        sememe_dict = {w:i + len(all_words) for i,w  in enumerate(set_sememe)}
        adj = np.eye(len(sememe_dict) + len(all_words), dtype=np.int)
        for i, word in enumerate(all_words):
            first_word = first_words[i]
            sememes = all_sememes[i]
            for first_w, sememe in zip(first_word, sememes):
                first_index = sememe_dict[first_w]
                sememe_index = [sememe_dict[w] for w in sememe]
                adj[i, first_index] = 1
                adj[first_index, i] = 1
                for j in sememe_index:
                    adj[j, first_index] = 1
                    adj[first_index, j] = 1
        sememe_indices = [self.sememes_dict[w] for w in sememe_dict]
        adj = sparse.csr_matrix(adj)
        return adj, sememe_indices

    def transform2indices(self, data, mode='train'):
        documents, source_labels, mention_set, part_of_speeches = data
        input_ids, input_masks, input_segments, input_labels, input_poses = [], [], [], [], []
        input_sememes, input_sememes_nums= [], []
        input_adjs = []
        label_dict = {'B':0, 'I':1, 'O':2}
        print("start... {}".format(mode))
        for document, source_label, source_pos in tqdm(zip(documents, source_labels, part_of_speeches), total=len(documents)):
            sentence_ids, sentence_masks, sentence_segments, sentence_labels, sentence_poses = [], [], [], [], []
            # 句子数，单词数 ,义原数
            sentence_sememes, sentence_sememes_nums = [], []
            global_max = 10
            tmp = [[w for line in document for w in line][:self.max_length - 2]]
            if self.args.doc == 'doc':
                self.max_length = 512
                indexs = self.get_sentence_index(document, self.max_length - 2)
                document = [[w for j in index for w in document[j]] for index in indexs]
                source_label = [[w for j in index for w in source_label[j]] for index in indexs]
                source_pos = [[w for j in index for w in source_pos[j]] for index in indexs]
            #nums = [self.get_sememe_and_num(w, 0)[1] for w in document]
            #max_nums = min(global_max, max(max(w) for w in nums))
            sentence_adj, sentence_sememes = self.get_sememe_adj(document)
                
            for sentence, s_label, pos in zip(document, source_label, source_pos):
                sememe, sememe_num = self.get_sememe_and_num(sentence, 0)
                #assert all(max_nums == len(w) for w in sememe)
                tokens = [self.CLS] + sentence + [self.SEP]
                input_pos = [self.CLS] + pos + [self.SEP]
                input_pos = [self.pos_dict[w if w in self.pos_dict else '[UNK]'] for w in input_pos]

                input_id  = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_id)
                input_segment = [0] * len(tokens)
                input_label = [2] + [label_dict[w] for w in s_label] + [2]

                padding = [0] * (self.max_length - len(input_id))

                input_id += padding
                input_mask += padding
                input_segment += padding
                input_label += padding
                input_pos += padding
                assert len(input_id) == self.max_length
                assert len(input_mask) == self.max_length
                assert len(input_segment) == self.max_length
                assert len(input_label) == self.max_length
                assert len(input_pos) == self.max_length
                assert len(sememe) == self.max_length

                sentence_ids.append(input_id)
                sentence_masks.append(input_mask)
                sentence_segments.append(input_segment)
                sentence_labels.append(input_label)
                sentence_poses.append(input_pos)
                #sentence_sememes.append(sememe)
                sentence_sememes_nums.append(sememe_num)

            input_ids.append(sentence_ids)
            input_masks.append(sentence_masks)
            input_segments.append(sentence_segments)
            input_labels.append(sentence_labels)
            input_poses.append(sentence_poses)
            input_sememes.append(sentence_sememes)
            #input_adjs.append(sentence_adj)
            #input_sememes_nums.append(sentence_sememes_nums)
            input_sememes_nums.append(sentence_adj)
        path = open(os.path.join(self.data_dir, '{}.pkl'.format(mode)), 'wb')
        static_matrix_path = os.path.join(self.data_dir, 'static_sememe_matrix.pkl')
        if os.path.exists(static_matrix_path):
            static_matrix = pkl.load(open(static_matrix_path, 'rb'))
            input_sememes, input_sememes_nums = static_matrix[mode]
        pkl.dump((input_ids, input_masks, input_segments, input_labels, mention_set, input_poses, input_sememes, input_sememes_nums), path)

    def manage(self):
        modes = ['train', 'test', 'valid']
        print("start building source file")
        train_data = self.execute('train')
        valid_data = self.execute('valid')
        test_data = self.execute('test')
        print("start building pos dict")
        self.build_label_dict(train_data[-1] + valid_data[-1] + test_data[-1])

        #self.build_dict(train_data, test_data, valid_data)
        #self.build_embedding(self.word_dict, '../data/embeddings/Tencent_AILab_ChineseEmbedding.txt')

        print("Start transforme to indices")
        self.get_sememe()

        self.transform2indices(train_data)
        self.transform2indices(test_data, 'test')
        self.transform2indices(valid_data, 'valid')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default='notest')
    parser.add_argument('--doc', default='doc')
    args = parser.parse_args()
    preprocessor = Preprocessor(args)
    preprocessor.manage()
