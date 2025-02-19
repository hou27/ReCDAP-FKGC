from data_loader import *
from params import *
from trainer import *
import json
import datetime

if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # if params['step'] == 'train' and params['checkpoint_state_dict_file'] is None:
    if params['step'] == 'train':
        params['prefix'] = params['prefix'] + '_' + str(params['seed']) + '_' + str(params['device']) + '_' + f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    elif params['step'] == 'test' and params['prefix'] == 'revised':
        params['prefix'] = params['prefix'] + '_' + str(params['dataset'])

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        dgl.random.seed(SEED)
        dgl.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path'] + v

    tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks' + tail]))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates' + tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2' + tail]))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    dataset['rel2id'] = json.load(open(data_dir['rel2ids']))

    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        dataset['ent2emb'] = np.loadtxt(params['data_path'] + '/entity2vec.TransE')
        dataset['rel2emb'] = np.loadtxt(params['data_path'] + '/relation2vec.TransE')

    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        trainer.eval(istest=True)
    elif params['step'] == 'dev':
        print(params['prefix'])
        trainer.eval(istest=False)
