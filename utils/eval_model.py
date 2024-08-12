import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score


def eval(model, device, have_prj, loader, metric_loss, miner, criterion, split):
    model.eval()
    print('Evaluating model on ' + split + ' data')
    birads_loss_sum = 0
    density_loss_sum = 0
    ce_loss_sum = 0
    metric_loss_sum = 0
    correct_birads = 0
    correct_density = 0
    f1_pred_birads = []
    f1_res_birads = []
    f1_pred_density = []
    f1_res_density = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images, labels = data
            label_birads = labels[0]
            label_density = labels[1]
            label_birads = label_birads - 1
            label_density = label_density - 1
            b = [item.item() for item in label_birads]
            c = [item.item() for item in label_density]
            f1_res_birads.extend(b)
            f1_res_density.extend(c)
            # print(a)
            images = images.to(device)
            label_birads = label_birads.to(device)
            label_density = label_density.to(device)
            if have_prj:
                p, logits = model(images)
                pminer = miner(p, labels)
                p_mloss = metric_loss(p, labels, pminer)
                ce_loss = criterion(logits, labels)
            else:
                _, pred_birad, pred_density = model(images)
                # p_mloss = torch.tensor([0.0])
                birads_loss = nn.CrossEntropyLoss()(pred_birad, label_birads.long())
                density_loss = nn.CrossEntropyLoss()(pred_density, label_density.long())
                ce_loss = birads_loss + density_loss
            birads_loss_sum += birads_loss.item()
            density_loss_sum += density_loss.item()
            ce_loss_sum += ce_loss.item()
            metric_loss_sum += 0

            pred1 = pred_birad.max(1, keepdim=True)[1]
            pred2 = pred_density.max(1, keepdim=True)[1]
            correct_birads += pred1.eq(label_birads.view_as(pred1)).sum().item()
            correct_density += pred2.eq(label_density.view_as(pred2)).sum().item()
            b = [item.item() for item in pred1]
            c = [item.item() for item in pred2]
            f1_pred_birads.extend(b)
            f1_pred_density.extend(c)

    f1_birads = f1_score(f1_res_birads, f1_pred_birads, average='macro')
    f1_density = f1_score(f1_res_density, f1_pred_density, average='macro')
    loss_avg = ce_loss_sum / (i+1)
    birads_loss_avg = birads_loss_sum / (i+1)
    density_loss_avg = density_loss_sum / (i+1)

    metric_loss_avg = metric_loss_sum / (i+1)

    accuracy_birads = correct_birads / len(loader.dataset)
    accuracy_density = correct_density / len(loader.dataset)

    return loss_avg, birads_loss_avg, density_loss_avg, accuracy_birads, accuracy_density, f1_birads, f1_density
