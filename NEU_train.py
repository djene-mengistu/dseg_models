#import libraries

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import time
import warnings
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from losses import*
from metrics import*
from NEU_dataloaders import*
from model import*
from NEU_utilities import get_logger, create_dir
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
seed = 1234
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# save_path = '/media/disk2t_/Dejene/DD/NEU-DualSeg/logs/'
class Trainer(object):
    '''This class takes care of training and validation of our model'''
    
    
    def __init__(self, model):
        self._init_logger()
        self.num_workers = 6
        self.patience = 0
        # self.best_dice = 0
        # self.best_loss_score = False
        self.batch_size = {"train": 16, "val": 16}
        self.accumulation_steps = 128 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 300
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = BCESoftDiceLoss(w1=0.4, w2=0.8)
        self.criterion = BCEDiceLoss()
        # self.criterion = FocalLoss()
        # self.criterion = DiceLoss()
        # self.optimizer = RAdam(self.net.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=10, verbose=True)
        self.net = self.net.to(self.device)
        # self.save_path = '/media/disk2t_/Dejene/DD/supervised_neu/ResT-small-FPNhead-notebook/Checkpoints/'
        cudnn.benchmark = True
        self.dataloaders = {
            phase: dataloaders(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        # self.save_tbx_log = self.save_path + '/tbx_logs'
        # self.writer = SummaryWriter(self.save_tbx_log)
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def _init_logger(self):

        log_dir = '.../model_weights/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        
        self.save_path = log_dir 

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)
    
    def iterate(self, epoch, phase):

        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
            
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)

        if phase == "train":

            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
            self.writer.add_scalar('Train/DSC', dice, epoch)
            self.writer.add_scalar('Train/IoU', iou, epoch)
        else:


            self.writer.add_scalar('Val/Loss', epoch_loss, epoch)
            self.writer.add_scalar('Val/DSC', dice, epoch)
            self.writer.add_scalar('Val/IoU', iou, epoch)
            # self.writer.add_scalar('Info/lr', lr_, epoch)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        
        # self.writer.add_scalar('Val_Dices', dice['val'], epoch)
        torch.cuda.empty_cache()
        return epoch_loss, dice

    def start(self):
        
        for epoch in range(0, self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss, dice = self.iterate(epoch, "val")
                # self.writer.add_scalar('Val/val_loss', val_loss, epoch)
                # self.scheduler.step(val_loss)
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
            # if self.best_dice < dice:    
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                Checkpoints_Path = self.save_path + '/Checkpoints'
                if not os.path.exists(Checkpoints_Path):
                    os.makedirs(Checkpoints_Path)
                torch.save(state, Checkpoints_Path + '/NEU_ResT_S_UperHead.pth')
                self.patience = 0
                # self.logger.info('current patience :{}'.format(self.patience))
            else:
                # self.save_best_model_1 = False
                self.patience += 1
                # self.logger.info('current patience :{}'.format(self.patience))
                    
            for param_group in self.optimizer.param_groups:
                lr_ = param_group['lr'] #For plotting the learning rate change during the training process
            
            self.writer.add_scalar('Info/lr', lr_, epoch)
            self.logger.info('current patience :{}'.format(self.patience))
            print('==================================================================================')
            print()

if __name__ == '__main__':
    # Training data path
    train_df_path = ".../data/DD/NEU_data/NEU_train_files.csv"
    data_folder = ".../data/NEU_data/"
    # test_data_folder = ".../data/NEU_data/test_set"
    model_trainer = Trainer(model)
    model_trainer.start()