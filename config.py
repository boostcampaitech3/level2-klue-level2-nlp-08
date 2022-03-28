import argparse
import json
import wandb
from collections import defaultdict

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', '-f', help='Config(Option) File', dest = 'config', default='myconfig.json')

    with open(parser.parse_args().config, 'r') as json_file:
        config_file = json.load(json_file)

    config = defaultdict(str)

    config['seed'] = config_file['seed']
    config['model_name'] = config_file['model']['name']
    config['tokenizer_name'] = config_file['tokenizer']['name']
    config['data_path'] = config_file['data']['path']
    config['project'] = config_file['wandb']['project']
    config['entity'] = config_file['wandb']['entity']
    config['name'] = config_file['wandb']['name'] + config_file['wandb']['special']
    config['optimizer'] = config_file['train']['optimizer']['type']
    config['lr'] = config_file['train']['optimizer']['args']['lr']
    config['scheduler'] = config_file['train']['scheduler']['type']
    config['num_warmup_steps'] = config_file['train']['scheduler']['num_warmup_steps']
    config['num_cycles']=config_file['train']['scheduler']['num_cycles']
    config['loss'] = config_file['train']['loss']['type']
    config['freezing_epochs'] = config_file['train']['freezing']['freezing_epochs']
    config['output_dir'] = config_file['train']['train_argument']['output_dir']
    config['save_total_limit'] = config_file['train']['train_argument']['save_total_limit']
    config['save_steps'] = config_file['train']['train_argument']['save_steps']
    config['num_train_epochs'] = config_file['train']['train_argument']['num_train_epochs']
    config['learning_rate'] = config_file['train']['train_argument']['learning_rate']
    config['per_device_train_batch_size'] = config_file['train']['train_argument']['per_device_train_batch_size']
    config['per_device_eval_batch_size'] = config_file['train']['train_argument']['per_device_eval_batch_size']
    config['warmup_steps'] = config_file['train']['train_argument']['warmup_steps']
    config['weight_decay'] = config_file['train']['train_argument']['weight_decay']
    config['logging_dir'] = config_file['train']['train_argument']['logging_dir']
    config['evaluation_strategy'] = config_file['train']['train_argument']['evaluation_strategy']
    config['eval_steps'] = config_file['train']['train_argument']['eval_steps']
    config['load_best_model_at_end'] = config_file['train']['train_argument']['load_best_model_at_end']
    config['report_to'] = config_file['train']['train_argument']['report_to']
    config['metric_for_best_model'] = config_file['train']['train_argument']['metric_for_best_model']
    config['fp16'] = config_file['train']['train_argument']['fp16']
    config['fp16_opt_level'] = config_file['train']['train_argument']['fp16_opt_level']

    print(config)




