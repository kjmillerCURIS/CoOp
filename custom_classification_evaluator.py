import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

class CustomClassificationEvaluator:
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname):
        self.cfg = cfg
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_group_res = defaultdict(list)
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._per_group_res = defaultdict(list)

    def process(self, mo, gt, dom):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        # dom (torch.LongTensor): domain index [batch] (need to know the domain-split in order to translate this into domain names)
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        
        for i, (domain, label) in enumerate(zip(dom, gt)):
            group = (domain.item(), label.item())
            matches_i = int(matches[i].item())
            self._per_group_res[group].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        groups = sorted(self._per_group_res.keys())

        print("=> per-group result")
        accs = []

        for group in groups:
            classname = self._lab2cname[group[1]]
            res = self._per_group_res[group]
            correct = sum(res)
            total = len(res)
            acc = 100.0 * correct / total
            accs.append(acc)
            print(
                f"* domain: {group[0]}\t"
                f"* class: {group[1]} ({classname})\t"
                f"total: {total:,}\t"
                f"correct: {correct:,}\t"
                f"acc: {acc:.1f}%"
            )
        mean_acc = np.mean(accs)
        print(f"* average: {mean_acc:.1f}%")

        results["pergroup_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
