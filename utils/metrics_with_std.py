import torch



import torch
from utils.confusion_matrix import confusion_matrix

AVAILABLE_METRICS = ['FP', 'FN', 'TP', 'TN', 'precision', 'acc', 'dice', 'specifity', 'iou', 'recall', 'mk', 'npv', 'mcc',
                     'bm', 'fnr', 'fpr', 'tpr', 'tnr', 'f1']


def calculate_single_metrics(single_matrix):
    FP = single_matrix.sum(0) - torch.diag(single_matrix)
    FN = single_matrix.sum(1) - torch.diag(single_matrix)
    TP = torch.diag(single_matrix)
    TN = single_matrix.sum() - (FP + FN + TP)
    precision = TP / (TP + FP)
    acc = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    npv = TN / (TN + FN)
    fnr = FN / (TP + FN)
    fpr = FP / (FP + TN)
    mcc = (TP * TN - FP * FN) / torch.sqrt((TP + FP).double() * (TP + FN) * (TN + FP) * (TN + FN))
    f1 = 2 * (precision * recall) / (precision + recall)
    specficity = TN / (TN + FP)
    iou = TP / (TP + FN + FP)
    dice = (2 * TP) / (2 * TP + FN + FP)
    return {'FP': FP, 'FN': FN, 'TP': TP, 'TN': TN, 'precision': precision, 'acc': acc, 'dice': dice,
            'specifity': specficity, 'iou': iou, 'recall': recall, 'mk': precision + npv - 1, 'npv': npv,
            'mcc': mcc, 'bm': (recall + specficity - 1), 'fnr': fnr, 'fpr': fpr, 'tpr': recall, 'tnr': specficity,
            'f1': f1}


class Metric:
    def __init__(self, num_classes, threshold=0.5):
        self.num_classes = num_classes
        self.confusion_dimension = 2 if self.num_classes == 1 else self.num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.matrix_list = []

    def update(self, output, target):
        if output.dim() == 4 and self.num_classes != 1:
            output = torch.max(output, dim=1)[1]

        target = target > 0.5
        if self.num_classes == 1:
            output = torch.where(output >= self.threshold, 1, 0)
        output = output.long()
        target = target.long()

        matrix = confusion_matrix(output, target, self.confusion_dimension)
        self.matrix_list.append(matrix.detach())

    def evaluate(self, require_single_results=False):
        result_list = [calculate_single_metrics(matrix) for matrix in self.matrix_list]
        sample_num = len(result_list)
        results_average = {}
        results_std = {}

        # print('shape', result_list[0]['dice'])

        if require_single_results:
            required_single_results = {}
            for metric_name in AVAILABLE_METRICS:
                required_single_results[metric_name] = [result[metric_name][1].item() for result in result_list]
            return required_single_results

        filtered_result_list = []
        for single_result in result_list:
            no_nan = True
            for metric_name in AVAILABLE_METRICS:
                if torch.isnan(single_result[metric_name]).any():
                    no_nan = False
                    break
            if no_nan:
                filtered_result_list.append(single_result)

        for metric_name in AVAILABLE_METRICS:
            results_average[metric_name] = sum([result[metric_name] for result in filtered_result_list]) / sample_num
            results_std[metric_name] = torch.std(torch.stack([result[metric_name].float() for result in filtered_result_list]), dim=0)
        return results_average, results_std