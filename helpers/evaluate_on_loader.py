from tqdm import tqdm
from utils.metrics import compute_confusion_matrix, iou_from_confusion, f1_from_confusion, precision_from_confusion, recall_from_confusion
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils.metrics import sigmoid_to_probs
import torch.nn as nn
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def evaluate_on_loader(model : nn.Module, dataloader : torch.utils.data.DataLoader, save_results_path : str, device : torch.device, threshold=0.5, save_pr_curve_path=None):
    model.eval()

    total_tp = total_fp = total_fn = total_tn = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating (test)")
        for batch in pbar:
            img_A = batch["image_A"].to(device)
            img_B = batch["image_B"].to(device)
            label = batch["label"].to(device)

            logits = model(img_A, img_B)
            probs = sigmoid_to_probs(logits)

            tp, fp, fn, tn = compute_confusion_matrix(logits, label, threshold=threshold)
            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            total_tn += tn.item()

            all_probs.append(probs.view(-1).cpu().numpy())
            all_targets.append(label.view(-1).cpu().numpy())

            #save result 
            os.makedirs(save_results_path, exist_ok=True)

            save_path = os.path.join(save_results_path, batch["filename"])
            pred_pil = TF.to_pil_image(probs)
            pred_pil.save(save_path)    

            

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    tp, fp, fn, tn = total_tp, total_fp, total_fn, total_tn

    iou = iou_from_confusion(tp, fp, fn)
    f1 = f1_from_confusion(tp, fp, fn)
    precision = precision_from_confusion(tp, fp)
    recall = recall_from_confusion(tp, fn)

    pr_precision, pr_recall, _ = precision_recall_curve(all_targets, all_probs)
    ap = average_precision_score(all_targets, all_probs)

    metrics = {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "ap": float(ap),
    }

    print("==== TEST RESULTS ====")
    print(f"IoU        : {metrics['iou']:.4f}")
    print(f"F1-score   : {metrics['f1']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"AP (PR AUC): {metrics['ap']:.4f}")
    print(f"TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, TN: {tn:.0f}")

    if save_pr_curve_path is not None:
        os.makedirs(os.path.dirname(save_pr_curve_path), exist_ok=True)
        plt.figure()
        plt.plot(pr_recall, pr_precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (AP={ap:.4f})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_pr_curve_path, dpi=300)
        plt.close()
        print(f"Saved PR curve to {save_pr_curve_path}")

    return metrics