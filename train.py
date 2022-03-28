from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
import wandb

from metric import *
from equip import *

import seaborn as sns

class MyTrainer(Trainer):
    def __init__(self, loss=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if loss is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # config에 저장된 loss_name에 따라 다른 loss 계산

        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = self.criterion(outputs[0], labels)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, optimizers):
        no_decay = ["bias", "LayerNorm.weight"]
        # Add any new parameters to optimize for here as a new dict in the list of dicts

        self.optimizer = optimizers[0]
        self.lr_scheduler = optimizers[1]

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)

        preds = output.predictions
        labels = output.label_ids
        self.draw_confusion_matrix(preds, labels)

        return output

    def draw_confusion_matrix(self, pred, label_ids):
        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cmn = cmn.astype('int')
        fig = plt.figure(figsize=(22, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        cm_plot = sns.heatmap(cm, cmap='Blues', fmt='d', annot=True, ax=ax1)
        cm_plot.set_xlabel('pred')
        cm_plot.set_ylabel('true')
        cm_plot.set_title('confusion matrix')
        cmn_plot = sns.heatmap(
            cmn, cmap='Blues', fmt='d', annot=True, ax=ax2)
        cmn_plot.set_xlabel('pred')
        cmn_plot.set_ylabel('true')
        cmn_plot.set_title('confusion matrix normalize')
        wandb.log({'confusion_matrix': wandb.Image(fig)})

def freezing(model, optimizer, loss_fn, freezing_epochs:int = 0, train_data=None):
    for param in model.parameters():
        param.requires_grad = False

    for params in model.classifier.parameters():
        params.requires_grad = True

    train_dataloader = DataLoader(train_data, batch_size=32, num_workers=5, shuffle=True)

    for epoch in range(freezing_epochs):
        model.train()
        for batch_idx, item in enumerate(train_dataloader):
            print(batch_idx, len(train_dataloader))
            labels = item.pop('labels')
            outputs = model(**item)
            optimizer.zero_grad()
            loss = loss_fn(outputs[0], labels)
            loss.backward()
            optimizer.step()

    for param in model.parameters():
        param.requires_grad = True

def train(config, tokenizer, RE_train_data, RE_test_data):
    ##############GET PRETRAINED MODEL###############
    model = get_model(config['model_name'], tokenizer)

    ##############GET EQUIP###############
    criterion = get_loss(config['loss'])
    optimizer = get_optimizer(model, config)
    num_train_steps = int(math.ceil(len(RE_train_data) / config['per_device_train_batch_size']))
    scheduler = get_scheduler(optimizer, config, num_train_steps = num_train_steps)
    optimizers = (optimizer, scheduler)

    ##############SETUP WANDB###############
    wandb.init(
        project=config['project'],
        entity=config['entity'],
        name=config['name']
    )

    ##############FREEZINNG###############
    freezing(model = model, optimizer = get_optimizer(model, config, freezing = True),
             loss_fn = criterion, freezing_epochs=config['freezing_epochs'],
             train_data=RE_train_data)

    ##############SETTING Argument###############
    training_args = TrainingArguments(
        output_dir=config['output_dir'],  # output directory
        save_total_limit=config['save_total_limit'],  # number of total save model.
        save_steps=config['save_steps'],  # model saving step.
        num_train_epochs=config['num_train_epochs'],  # total number of training epochs
        learning_rate=config['learning_rate'],  # learning_rate
        per_device_train_batch_size=config['per_device_train_batch_size'],  # batch size per device during training
        per_device_eval_batch_size=config['per_device_eval_batch_size'],  # batch size for evaluation
        #  warmup_steps= config['warmup_steps'],  # number of warmup steps for learning rate scheduler
        weight_decay=config['weight_decay'],  # strength of weight decay
        logging_dir=config['logging_dir'],  # directory for storing logs
        logging_steps=config['logging_steps'],  # log saving step.
        evaluation_strategy=config['evaluation_strategy'],  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=config['eval_steps'],  # evaluation step.
        load_best_model_at_end=config['load_best_mode_at_end'],
        metric_for_best_model = config['metric_for_best_model'],
        report_to = config['report_to'],
        fp16=config['fp16'],
        fp16_opt_level=config['fp16_opt_level']
    )

    ##############Training###############
    trainer = MyTrainer(
        loss = criterion,
        model = model,
        args = training_args,
        train_dataset = RE_train_data,
        eval_dataset = RE_test_data,
        compute_metrics = compute_metrics,
    )
    if config['change_optimizer']:
        trainer.create_optimizer_and_scheduler(optimizers)

    trainer.train()
    model.save_pretrained('./best_model')