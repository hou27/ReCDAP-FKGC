import random
import numpy as np
import math

class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.curr_rel_idx = 0
        self.tasks = dataset[step + '_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']
        self.all_ents = list(dataset['ent2id'].keys())

        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

    #     # 전체 배치 수 계산
    #     if step == 'train':
    #         # 한 에폭당 모든 관계를 한 번씩 사용할 때 배치 수 계산
    #         self.total_batches = math.ceil(self.num_rels / self.bs)
    #     else:
    #         # 평가 단계의 경우 전체 트리플을 배치 크기로 나눔
    #         self.total_batches = math.ceil(self.num_tris / self.bs)

    # def __len__(self):
    #     return self.total_batches

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        random_idx = random.sample(np.arange(0, len(curr_tasks), 1).tolist(), self.few + self.nq)
        support_triples = [curr_tasks[i] for i in random_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in random_idx[self.few:]]

        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            support_negative_triples.append([e1, rel, negative])
        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            negative_triples.append([e1, rel, negative])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
        if len(dis_joint) == 0:
            dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            dis_joint = random.sample(dis_joint, 10)
        for negative in dis_joint:
            negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel
