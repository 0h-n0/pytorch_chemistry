import time
import json
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable

from e2edd.utils.mylogger import mylogger
from pytorch_chemistry.models import Accuracy
from pytorch_chemistry.models import AucRocAccuracy
from pytorch_chemistry.models import SigmoidCrossEntropyLoss


class Classifier(nn.Module):
    def __init__(self, model, criterion):
        super(Classifier, self).__init__()
        self.model = model
        self.criterion = criterion

    def __call__(self, x1, x2, t):
        out = self.model(x1, x2)
        loss = self.criterion(out, t)
        return loss.mean()
    
    
class Trainer(object):
    def __init__(
            self,
            *,
            model: torch.nn.Module,
            optimizer,
            train_iterator,
            cv_iterator,
            test_iterator,
            epochs=100,
            log_interval=10,                 
            benchmark_mode=True,
            ngpu=1,
            workdir='train',
            label_names=None,
            ):
        self.model = model
        self.optimizer = optimizer
        self.train_iterator = train_iterator
        self.cv_iterator = cv_iterator
        self.test_iterator = test_iterator                    
        self.epochs = epochs
        self.log_interval = log_interval
        self.ngpu = ngpu
        self.workdir = Path(workdir)
        self.epoch_total_time = None
        self.gradient_clipping = False
        self.gradient_clipping_max_value = 2
        self.label_names = label_names
        
        if benchmark_mode:
            torch.backends.cudnn.benchmark = benchmark_mode
        mylogger.debug(f'benchmark mode = {benchmark_mode}')
        mylogger.debug(f'self.ngpu = {self.ngpu}')
        
        self.criterion = SigmoidCrossEntropyLoss()
        self.acc = Accuracy()
        #self.acc = AucRocAccuracy()
        
        if self.ngpu == 1:
            self.cuda()
            self.criterion.cuda()
        elif self.ngpu > 1:
            gpus = [i for i in range(self.ngpu)]
            self.model.cuda()
            self.model = torch.nn.DataParallel(Classifier(self.model,
                                                          self.criterion),
                                               device_ids=gpus)
            self.cuda()
        else:
            self.cuda(False)
        self.workdir.mkdir(exist_ok=True)        

    def cuda(self, flag=False):
        if flag:
            self.model.cuda()
            self.gpu = flag            
        else:
            self.model.cpu()
            self.gpu = flag

    def run(self, start_epoch=0, dry_run=False):
        tr_path = self.workdir / 'train_loss.json'
        tr_report = {}        
        cv_path = self.workdir / 'cv_loss.json'
        cv_report = {}
        
        for i in range(start_epoch, self.epochs):
            mylogger.debug(f'ecpoh = {i}')
            if self.epoch_total_time is None:
                s_epoch_total_time = time.perf_counter()
            self._run(i, self.train_iterator, train=True,
                      output_path=tr_path, report=tr_report, dry_run=dry_run)
            if not dry_run:
                self._run(i, self.cv_iterator, train=False,
                          output_path=cv_path, report=cv_report, dry_run=dry_run)
            if self.epoch_total_time is None:
                self.epoch_total_time = time.perf_counter() - s_epoch_total_time


    def _run(self, epoch, iterator, train, output_path=None,
             report={}, dry_run=False):
        if train:
            self.model.train()
            mode = 'Train'
        else:
            self.model.eval()
            mode = 'CV'
            
        sum_loss = 0
        total_time = 0
        num_iterations = len(iterator)
        total_preprocess_time = 0
        start_preprocess_time = None
        for i, batch in enumerate(iterator):
            if start_preprocess_time is not None:
                total_preprocess_time += time.perf_counter() - start_preprocess_time
            mylogger.debug(f"{epoch}th epoch, {i}th iteration start:"
                           "---------------------------------------")
            start_time = time.perf_counter()
            loss, acc = self.update(batch, train)
            sum_loss += loss
            total_time += time.perf_counter() - start_time
            if i % self.log_interval == 0:
                self.console_log(mode, epoch, num_iterations, i,
                                 start_time, start_preprocess_time,
                                 sum_loss, total_time, total_preprocess_time,
                                 acc)

            if i == 0 and mode == 'CV':
                self.decode_data = batch
            if dry_run:
                break
            start_preprocess_time = time.perf_counter()
            
        if output_path:
            fp = output_path.open('w')
            _time = time.time() - total_time            
            report['{:03d}'.format(epoch)] = dict(
                mode=mode,
                loss=float(sum_loss) / (i+1),
                acc=acc.tolist(),
                time=_time,
                daytime=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            )
            json.dump(report, fp)
            fp.close()

        if not dry_run and mode == 'Train':
            self.save_model('model{:03d}.pt'.format(epoch))
            if self.gpu:
                self.model.cuda()

    def console_log(self, mode, epoch, num_iterations, i, start_time,
                    start_preprocess_time, sum_loss, total_time,
                    total_preprocess_time, acc):
        average_train_time = total_time / (i + 1)
        average_preprocess_time = total_preprocess_time / (i + 1)
        average_time = average_train_time + average_preprocess_time
        residual_time = average_time * (num_iterations - (i + 1))
        hour = residual_time // 3600
        minute = (residual_time % 3600) // 60
        second = (residual_time % 60)
        if self.epoch_total_time is not None:
            total_training_time = (self.epochs - epoch) * self.epoch_total_time
            tday = total_training_time // 3600 // 24
            thour = total_training_time // 3600 % 24
            tminute = (total_training_time % 3600) // 60
        else:
            total_training_time = 0
            tday = total_training_time // 3600 // 24
            thour = total_training_time // 3600 % 24
            tminute = (total_training_time % 3600) // 60
        if start_preprocess_time is None:
            preprocess_time = 0
        else:
            preprocess_time = time.perf_counter() - start_preprocess_time
        mylogger.info(f'*****   {mode:5s}: epoch {epoch:3d}/{self.epochs:3d}   ******')            
        _format1 = (':LOSS{:>7.4f}: {:7.4f} ({:7.4f}:preprocess) '
                    'sec/batch: {:3d}/{:3d} batchs')
        _format2 = ('Est. Epoch:{:2d}h {:2d}m {:5.2f}s:'
                    ' Est. Total:{:2d}d {:2d}h {:2d}m')

        mylogger.info(_format1.format(
            float(sum_loss) / (i+1),
            time.perf_counter() - start_time + preprocess_time,
            preprocess_time,
            i, num_iterations))
        mylogger.info(_format2.format(
            int(hour), int(minute), second,
            int(tday), int(thour), int(tminute)
        ))

        if self.label_names:
            _format3 = ''
            for l in self.label_names:
                _format3 += f'{l}:{{:5.1f}}% '
            mylogger.info(_format3.format(
                *acc
            ))
                
    def update(self, batch, train):
        mylogger.debug(f'in trainer.update: START: self.optimizer.zero_grad()')                
        self.optimizer.zero_grad()
        mylogger.debug(f'in trainer.update: END: self.optimizer.zero_grad()')

        x1 = batch['atoms'].long()
        x2 = batch['adj_matrix']
        t = torch.autograd.Variable(batch['assey'].float())

        if not isinstance(x1, Variable):
            x1 = Variable(x1)
        if not isinstance(x2, Variable):
            x2 = Variable(x2)            
        if not isinstance(t, Variable):
            t = Variable(t)            
            
        if self.ngpu <= 1:
            x = self.model(x1, x2)
            loss = self.criterion(x, t)
            acc = self.acc(x, t)
            params = self.model.parameters()
        elif self.ngpu > 1:
            loss = self.model(x1, x2, t)
            params = self.model.model.parameters()

        if self.gradient_clipping:
            total_norm = nn.utils.clip_grad_norm(
                params, self.gradient_clipping_max_value)
            mylogger.debug(f'gradinet clipping: total_norm = {total_norm}')
            
        if train:
            loss.backward()
            self.optimizer.step()
        mylogger.debug(f'in trainer.update Loss: {float(loss.data.cpu().numpy())}')
        return loss.data.cpu().numpy(), acc.data.cpu().numpy()        

    def save_model(self, name):
        outpath = self.workdir / name
        mylogger.info(f'save {str(outpath.resolve())}')
        if self.ngpu > 1:
             torch.save(self.model.cpu().state_dict(), str(outpath.resolve()))
        else:
             torch.save(self.model.state_dict(), str(outpath.resolve()))          
