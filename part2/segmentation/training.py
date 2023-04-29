import argparse
import datetime
import logging
import sys
import os
import shutil
import hashlib

import numpy as np
import torch.nn as nn
import torch.cuda
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../')
from part2.segmentation.model import UNetWrapper, SegmentationAugmentation
from part2.segmentation.dsets import TrainingLuna2dSegmentationDataset, Luna2dSegmentationDataset
from part2.util.util import enumerateWithEstimate

log = logging.getLogger(__name__)

METRICS_SIZE = 10
METRICS_LOSS_NDX = 1
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_workers', help='Number of worker processes for background data loading', default=8,
                            type=int)
        parser.add_argument('--batch_size', help='Batch size to use for training', default=32, type=int)
        parser.add_argument('--epoch', help='Number of epochs to train for', default=1, type=int)
        # TODO
        parser.add_argument('--tb_prefix', help='Data prefix to use for Tensorboard run.',
                            default='classification')
        parser.add_argument('--comment', help='Comment suffix for Tensorboard run.', nargs='?', default='dwlpt')
        parser.add_argument('--balanced', help="Balance the training data to half positive, half negative.",
                            action='store_true', default=False, )
        parser.add_argument('--augmented', help="Augment the training data.", action='store_true', default=False, )
        parser.add_argument('--augment-flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true', default=False, )
        parser.add_argument('--augment-offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true', default=False, )
        parser.add_argument('--augment-scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true', default=False, )
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true', default=False, )
        parser.add_argument('--augment-noise', help="Augment the training data by randomly adding noise to the data.",
                            action='store_true', default=False, )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        # model = LunaModel()

        segmentation_model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True,
                                         up_mode='upconv')
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
        if self.use_cuda:
            log.info("Using CUDA:{} device.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                #     model = nn.DataParallel(model)
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            # model = model.to(self.device)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)
        return segmentation_model, augmentation_model

    def initOptimizer(self):
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return Adam(self.segmentation_model.parameters())

    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(val_stride=10, isValSet_bool=True, contextSlices_count=3)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda)

        return val_dl

    def doTraining(self, epoch_ndx, train_dl):
        self.segmentation_model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        batch_iter = enumerateWithEstimate(train_dl, "E{} Training".format(epoch_ndx), start_ndx=train_dl.num_workers)
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.augmentation_model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, classificationThreshold=0.5):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        # logits_g, probability_g = self.model(input_g)
        prediction_g = self.segmentation_model(input_g)

        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32)
            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1, 2, 3])
        dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
    ):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
                                                   / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_NDX] \
                                             / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
                                      / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'data-unversioned', 'part2', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = 5
        for epoch_ndx in range(1, self.cli_args.epoch + 1):
            log.info(
                "Epoch {} of {}, {}/{} batches of size {}*{}".format(epoch_ndx, self.cli_args.epoch, len(train_dl),
                                                                     len(val_dl), self.cli_args.batch_size, (
                                                                         torch.cuda.device_count() if self.use_cuda else 1)))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)
                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logMetrics(epoch_ndx, 'trn', train_dl)
                self.logMetrics(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()


if __name__ == '__main__':
    SegmentationTrainingApp().main()
