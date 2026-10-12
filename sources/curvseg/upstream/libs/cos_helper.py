import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter, IOUMeter
from libs.postprocess import PostProcessor
from tqdm import tqdm
from prompt.tools import (segment_video_labels, gen_label, gen_label_split,
                          generate_segment_features,generate_split_features,
                          create_logits, split_feature, split_gt, split_gt_feature, split_mixed_class)
from prompt.text_prompt import text_prompt_for_clip
from libs.loss_fn.curve import CurvatureLoss
from libs.loss_fn.circle import CircleLoss, convert_label_to_similarity

from libs.loss_fn.rdrop import RDroLoss, CrossConstraintLoss
from libs.loss_fn.cross import TSConstraintLoss, RobustTSConstraint, KLTSConstraint

from libs.tsne import plot_tsne
import copy
import pandas as pd
import json
from libs.perturb import batch_probability_curvature_icd
# from libs.perturb_plot import batch_probability_curvature_icd_and_plot

curve_generator = CurvatureLoss(q=10)
# rdrop_criterion = RDroLoss()
# cross_criterion = RobustTSConstraint() #l1
# cross_criterion = KLTSConstraint()
# cross_criterion = TSConstraintLoss() #ce
circle_loss = CircleLoss(m=0.25, gamma=1)
perturb_eval = False
examplers = None
examplers_batch = 400
probcurve_batch = 800
curvature_batch = 0
pcurv_dir = '/home/dxr/LaSA/perturb_check'

def train(
    train_loader: DataLoader,
    model: nn.Module,
    model_text: nn.Module,
    class_text_list,
    joint_text_list,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    criterion_contrast: nn.Module,
    lambda_bound_loss: float,
    optimizer: optim.Optimizer,
    dataset_name,
    device, output_device,
    criterion_curve = None,
    batch_cnt = 0,
    alpha = 1.0,
    beta = 0.5,
    cross_criterion = TSConstraintLoss(),
) -> float:
    global examplers
    losses = AverageMeter("Loss", ":.4e")
    cross_losses = AverageMeter("Cross Loss", ":.4e")
    div_loss = AverageMeter("Div Loss", ":.4e")
    # switch training mode
    model.train()

    for sample in tqdm(train_loader):
        loss = 0.0
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]
        # print(model.action_decoder.query_embedding.weight)


        x = x.to(output_device)
        t = t.to(output_device)
        b = b.to(output_device)
        mask = mask.to(output_device)
        joint_text_list = joint_text_list.to(output_device)

        optimizer.zero_grad()

        batch_size = x.shape[0]
        joint_text_embedding = None
        if hasattr(model, 'text_token') != True or model.text_token == None:
            joint_text_embedding = model_text(joint_text_list).float()
            output_cls, output_bound, output_feature, output_feature_split, logit_scale = model(x, mask, joint_text_embedding)
            
        # compute output and loss
        else:
            output_cls, output_bound, output_feature, output_feature_split, logit_scale, dloss = model(x, mask, joint_text_embedding)
            loss += dloss
        #Action-text pairs
        t_segment = segment_video_labels(t)

        label =  [i[0] for seg in t_segment for i in seg]

        label_g = gen_label(label)
        num_class = len(class_text_list)
        texts = list()
        for single_label in label:
            text_item = class_text_list[single_label].unsqueeze(dim=0)
            texts.append(text_item)

        texts = torch.cat(texts).cuda(output_device)
        text_embedding = model_text(texts).float()
        

        texts_dxr = list(class_text_list[c].unsqueeze(dim=0) for c in range(num_class))
        texts_dxr = torch.cat(texts_dxr).cuda(output_device)
        text_embedding_dxr = model_text(texts_dxr).float()

        action_embeddings = []
        if examplers == None:
            examplers = [{c:(0.0, torch.rand_like(text_embedding_dxr[0])) for c in range(num_class)} for _ in range(len(output_feature))]
        d_bars = []
        if isinstance(output_feature, list):
            for i in range(len(output_feature)):
                
                action_embedding = generate_segment_features(output_feature[i], t_segment, output_device)
                action_embeddings.append(action_embedding)
                if batch_cnt >= examplers_batch:
                    d_bar = batch_probability_curvature_icd(output_feature[i].permute(0, 2, 1), \
                                                            t, text_embedding_dxr, examplers[i], mask,\
                                                            alpha=alpha, bate=beta)
                    # d_bar = batch_probability_curvature_icd_and_plot(output_feature[i].permute(0, 2, 1), t, text_embedding_dxr, examplers[i], mask, save_dir=pcurv_dir+'/'+f'{batch_cnt}')
                    d_bars.append(d_bar)
                    # 假设 d_bar: [B, T], t: [B, T], num_class 已知
                    B, T = d_bar.shape
                    d_flat = d_bar.reshape(-1)       # [B*T]
                    t_flat = t.reshape(-1)           # [B*T]
                    # print(d_bar.max(), d_bar.min())
                    for c in range(num_class):
                        # 找到类 c 的所有位置
                        mask_d = (t_flat == c)
                        if mask_d.sum() == 0:
                            continue

                        # 其余位置填 -inf，保证 argmax 只搜同类
                        neg_inf = torch.tensor(-float('inf'), device=d_flat.device)
                        d_masked = torch.where(mask_d, d_flat, neg_inf)  # [B*T]

                        # 全局 argmax
                        max_flat_idx = torch.argmax(d_masked)         # scalar idx in [0, B*T)
                        max_value    = d_masked[max_flat_idx]         # scalar Tensor

                        # 转回 2D 索引
                        b_idx = int(max_flat_idx // T)
                        t_idx = int(max_flat_idx %  T)

                        # 校验（用近似对比）
                        value_check = d_bar[b_idx, t_idx]
                        assert torch.isclose(max_value, value_check, atol=1e-6), (
                            f"class {c} mismatch: {max_value.item():.6f} vs {value_check.item():.6f}"
                        )

                        # 如果超过历史 best，更新 exampler
                        if max_value.item() <= examplers[i][c][0]:
                            feat_vec = output_feature[i].permute(0,2,1)[b_idx, t_idx]
                            examplers[i][c] = (max_value.item(), feat_vec.detach().clone())
                    # # print(output_feature[i].shape, t.shape, text_embedding_dxr.shape)
                    # # torch.Size([3, 512, 6000]) torch.Size([3, 6000]) torch.Size([8, 512])
                    
                    # # print(d_bar.shape)  # torch.Size([3, 6000])
                    # for c in range(num_class):
                    #     cls_d_bar = d_bar[t == c]
                    #     if cls_d_bar.shape[0] == 0:
                    #         continue
                    #     cls_idx = torch.where(t == c)
                    #     cls_idx = torch.stack(cls_idx, dim=1)
                    #     max_idx = torch.argmax(cls_d_bar)
                    #     max_value = cls_d_bar[max_idx]
                    #     max_cls_idx = cls_idx[max_idx]
                    #     b_idx, t_idx = max_cls_idx.tolist()      # 先把 tensor([b,t]) 拆成 Python int
                    #     print(max_idx, max_value, max_cls_idx, d_bar[b_idx, t_idx])
                    #     assert torch.isclose(max_value, d_bar[b_idx, t_idx]), "max_value != d_bar[max_cls_idx]"
                    #     if max_value >= examplers[i][c][0]:
                    #         examplers[i][c] = (max_value, output_feature[i].permute(0, 2, 1)[b_idx, t_idx].detach().clone())
                    # 


        #Clip-text pairs
        gt_split, feature_split = split_mixed_class(t_segment,2)

        feature_split_embedding = generate_split_features(output_feature_split, feature_split, output_device)

        text_split = text_prompt_for_clip(gt_split, dataset_name, "simple").cuda(output_device)

        text_split_embedding = model_text(text_split).float()

        label_split_g = gen_label_split(gt_split)

        # _, curve, _ = curve_generator.curvature_estimation(output_feature)


        
        # Action segmentation loss
        if isinstance(output_cls, list):
            n = len(output_cls)
            # print(n)
            for out in output_cls:
                # print(out, t, x)
                # print(criterion_cls(out, t, x))
                if batch_cnt >= probcurve_batch:
                    # print(out.shape, t.shape, x.shape)
                    loss += criterion_cls(out, t, x, frame_weights=d_bars[i]) / n
                else:
                    loss += criterion_cls(out, t, x) / n
        else:
            loss += criterion_cls(output_cls, t, x)

        # boundary regression loss
        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        # action-text contrastive loss
        if isinstance(action_embeddings, list):
            for i in range(len(action_embeddings)):
                logits_per_image, logits_per_text = create_logits(action_embeddings[i], text_embedding, logit_scale[0])
                ground_truth = torch.tensor(label_g, dtype=action_embedding.dtype, device=output_device)

                loss_imgs = criterion_contrast(logits_per_image, ground_truth)
                loss_texts = criterion_contrast(logits_per_text, ground_truth)

                loss += 0.8 * ((loss_imgs + loss_texts) / 2)

        # clip-text contrastive loss
        logits_per_image, logits_per_text = create_logits(feature_split_embedding, text_split_embedding,
                                                          logit_scale[1])
        ground_truth = torch.tensor(label_split_g, dtype=feature_split_embedding.dtype, device=output_device)

        loss_imgs = criterion_contrast(logits_per_image, ground_truth)
        loss_texts = criterion_contrast(logits_per_text, ground_truth)

        loss += 0.5 * ((loss_imgs + loss_texts) / 2)

        # curvature loss
        # if criterion_curve is not None:
        #     if isinstance(output_feature, list):
        #         n = len(output_feature)
        #         for i in range(len(output_feature)):
                    
        #             _, curve, _ = curve_generator.curvature_estimation(output_feature[i])
        #             # print(curve.shape, b.shape, mask.shape)
        #             # exit()
        #             loss += criterion_curve(curve.unsqueeze(1), b, mask) / n

        #     else:
        #         _, curve, _ = curve_generator.curvature_estimation(output_feature)
        #         loss += criterion_curve(curve.unsqueeze(1), b, mask)
        # div loss
        
        if model.div_loss is not None:
            loss += model.div_loss
            div_loss.update(model.div_loss, batch_size)
            # print("div loss: ", model.div_loss)
        # R-Drop loss
        # if isinstance(output_feature, list):
        #     n = len(output_feature)
        #     for i in range(len(output_feature)):
        #         _, curve, _ = curve_generator.curvature_estimation(output_feature[i])
        #         if isinstance(output_bound, list):
        #             m = len(output_bound)
        #             for j in range(len(output_bound)):
        #                 loss += rdrop_criterion(curve.unsqueeze(1), output_bound[j], mask) / (n*m)
        #         else:
        #             loss += rdrop_criterion(curve.unsqueeze(1), output_bound, mask) / n
        # else:
        #     _, curve, _ = curve_generator.curvature_estimation(output_feature)
        #     loss += rdrop_criterion(curve.unsqueeze(1), output_bound, mask)

        #CROSS CONSTRAINT LOSS
        # print("other loss: ", loss.item())
        # deep copy other loss
        if batch_cnt >= curvature_batch:
            cross_loss = loss.item()
            if isinstance(output_feature, list):
                n = len(output_feature)
                for i in range(len(output_feature)):
                    # _, curve, _ = curve_generator.curvature_estimation(output_feature[i])
                    curve = curve_generator._compute_step_distances(
                        output_feature[i].permute(0, 2, 1), step=1, metric='cosine'
                    )
                    if isinstance(output_bound, list):
                        m = len(output_bound)
                        for j in range(len(output_bound)):
                            # wo true boundary
                            # loss += cross_criterion(curve.unsqueeze(1), output_bound[j], mask) / (n*m)
                            # w true boundary
                            loss += cross_criterion(curve.unsqueeze(1), output_bound[j], mask, b) / (n*m)
                    else:
                        # wo true boundary
                        # loss += cross_criterion(curve.unsqueeze(1), output_bound, mask) / n
                        # w true boundary
                        loss += cross_criterion(curve.unsqueeze(1), output_bound, mask, b) / n

            else:
                # _, curve, _ = curve_generator.curvature_estimation(output_feature)
                curve = curve_generator._compute_step_distances(
                        output_feature.permute(0, 2, 1), step=1, metric='cosine'
                    )
                # wo true boundary
                # loss += cross_criterion(curve.unsqueeze(1), output_bound, mask)
                # w true boundary
                loss += cross_criterion(curve.unsqueeze(1), output_bound, mask, b)
            cross_loss = loss.item() - cross_loss
            cross_losses.update(cross_loss, batch_size)
        # circle loss
        # if isinstance(output_feature, list):
        #     n = len(output_feature)
        #     # print(t.shape)
        #     # exit()
        #     for i in range(len(output_feature)):
        #         N, C, T = output_feature[i].shape
        #         # print(N, C, T)
        #         output_feature_circle = output_feature[i].transpose(1, 2).reshape(N*T, C)
        #         t_circle = t.reshape(N*T, )
        #         pos, neg = convert_label_to_similarity(output_feature_circle, t_circle)
        #         loss += circle_loss(pos, neg) / n
        # else:
        #     N, C, T = output_feature.shape
        #     output_feature_circle = output_feature.transpose(1, 2).reshape(N*T, C)
        #     t_circle = t.reshape(N*T, )
        #     pos, neg = convert_label_to_similarity(output_feature_circle, t_circle)
        #     loss += circle_loss(pos, neg)
        # record loss
        losses.update(loss.item(), batch_size)


        loss.backward()
        optimizer.step()
    print("cross loss: ", cross_losses.avg)
    print("div loss: ", div_loss.avg)
    return losses.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    model_text: nn.Module,
    joint_text_list,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    device,output_device,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    boundary_th: float,
    tolerance: int,
    refinement_method: Optional[str] = None
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    losses = AverageMeter("Loss", ":.4e")
    postprocessor = PostProcessor(refinement_method, boundary_th)
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    # query_scores_cls = ScoreMeter(
    #     id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
    #     iou_thresholds=iou_thresholds,
    # )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(output_device)
            t = t.to(output_device)
            b = b.to(output_device)
            mask = mask.to(output_device)
            joint_text_list = joint_text_list.to(output_device)

            batch_size = x.shape[0]

            joint_text_embedding = model_text(joint_text_list).float()

            # compute output and loss
            output_cls, output_bound, _, _ = model(x, mask, joint_text_embedding)
            # print(query_cls)

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            ) #加上了边界的预测
            # update score
            scores_cls.update(output_cls, t, output_bound, mask) #The result of not utilizing boundary branch
            # query_scores_cls.update(query_cls.to("cpu").data.numpy(), t) #The result of utilizing boundary branch
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t) #The result of utilizing boundary branch

    before_refinement_cls_acc, before_refinement_edit_score, before_refinement_segment_f1s = scores_cls.get_scores()
    cls_acc, edit_score, segment_f1s = scores_after_refinement.get_scores()
    # query_cls_acc, query_edit_score, query_segment_f1s = query_scores_cls.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    print("Before refinement:", before_refinement_cls_acc, before_refinement_edit_score, before_refinement_segment_f1s)
    print("Boundary scores:", bound_acc, precision, recall, bound_f1s)
    # print("Query scores:", query_cls_acc, query_edit_score, query_segment_f1s)

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s,
        bound_acc,
        precision,
        recall,
        bound_f1s,
    )

def evaluate_(
    val_loader: DataLoader,
    model: nn.Module,
    model_text,
    joint_text_list,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    config : str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)
            joint_text_list = joint_text_list.to(device)

            joint_text_embedding = model_text(joint_text_list).float()

            # compute output and loss
            output_cls, output_bound = model(x, mask, joint_text_embedding)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)
            
    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )
    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))
    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )

from libs.measures import curvature_estimation
import matplotlib.pyplot as plt
import os
import numpy as np

def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    model_text,
    joint_text_list,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    config: str,
    refinement_method: Optional[str] = None,
    save_curvature: bool = False,
    save_stne: bool = False,
    action_text_list = None,
) -> None:
    plt.switch_backend('Agg')  # Use Agg backend for headless environments
    id2class_map = get_id2class_map(dataset, dataset_dir=dataset_dir)
    # print("id2class_map: ", id2class_map)
    # exit()
    postprocessor = PostProcessor(refinement_method, boundary_th)
    scores_before_refinement = ScoreMeter(
        id2class_map=id2class_map,
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )
    scores_after_refinement = ScoreMeter(
        id2class_map=id2class_map,
        iou_thresholds=iou_thresholds,
    )
    IOU_before_refinement = IOUMeter(
        id2class_map=id2class_map,
        max_len=9000,
    )
    IOU_after_refinement = IOUMeter(
        id2class_map=id2class_map,
        max_len=9000,
    )
    model.eval()
    # test = 7
    if perturb_eval:
        perturb_csv = '/DUANXINRAN/LaSA/csv/LARA/perturb_logs/boundary_perturb_log_split1.csv'
        perturb_df = pd.read_csv(perturb_csv)
        # 建立 feature_path -> (orig, new) 映射
        pert_map = {
            row['feature_path']: (int(row['orig_boundary']), int(row['new_boundary']))
            for _, row in perturb_df.iterrows()
        }
        pert_paths = set(pert_map.keys())
        # print(pert_map)
        
    # print(pert_paths)
    # exit()
    save_tsne_path = os.path.join(result_path, "tsne")
    if not os.path.exists(save_tsne_path):
        os.makedirs(save_tsne_path)
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(val_loader)):
            # if batch_idx == test:
            #     break
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]
            path = sample['feature_path'][0]
            if perturb_eval:
                if path not in pert_paths:
                    continue
            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            if perturb_eval:
                # print(path)
                orig_b, pert_b = pert_map[path]
                orig_b, pert_b = orig_b//4, pert_b//4
                # exit()
            
            mask = mask.to(device)
            joint_text_list = joint_text_list.to(device)

            joint_text_embedding = model_text(joint_text_list).float()
            action_text_embedding = None
            if action_text_list is not None:
                action_text_list = action_text_list.to(device)
                action_text_embedding = model_text(action_text_list).float()
            output_cls, output_bound, output_feature, bound_feature = model(x, mask, joint_text_embedding)
            # print("output_cls: ", output_cls.shape)
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()
            x_np = x.to("cpu").data.numpy()
            t_np = t.to("cpu").data.numpy()
            b_np = b.to("cpu").data.numpy()
            mask_np = mask.to("cpu").data.numpy()
            feature_np = output_feature.to("cpu").data.numpy().transpose(0, 2, 1)
            feature_split_np = bound_feature.to("cpu").data.numpy().transpose(0, 2, 1)
            output_bound_np = output_bound.transpose(0, 2, 1)
            curv_np = []

            # print(output_feature.shape)
            # print(t.shape)
            N, C, T = output_feature.shape
            output_feature_circle = output_feature.transpose(1, 2).reshape(N*T, C)
            t_circle = t.reshape(N*T, )
            if save_stne:
                plot_tsne(output_feature_circle, t_circle, os.path.join(save_tsne_path, f"tsne_action_{batch_idx}.png"))
            for sample_idx in range(x.size(0)):
                # seq_len from valid frames in this sample
                seq_len = int(mask_np[sample_idx].sum())
                # cur_feature: per-frame classification feature (n_frames, n_features)
                if perturb_eval:
                    l_b, r_b = max(min(orig_b, pert_b)-10, 0), min(max(orig_b, pert_b)+10, seq_len)
                    
                    cur_feature = torch.from_numpy(feature_np[sample_idx][l_b:r_b])
                    gt_cls = t_np[sample_idx][l_b:r_b].astype(int)
                    gt_br = b_np[sample_idx][:, l_b:r_b].astype(int)
                    # print(cur_feature.shape, feature_np.shape, t_np.shape, b_np.shape)
                    # print(gt_br, orig_b, pert_b, l_b, r_b, gt_cls)
                    # # print(t_np[sample_idx][636])
                    # exit()
                    pred_cls = np.argmax(output_cls[sample_idx][:, l_b:r_b],axis=0)
                else:

                    cur_feature = torch.from_numpy(feature_np[sample_idx][:seq_len])
                    gt_cls = t_np[sample_idx][:seq_len].astype(int)
                    pred_cls = np.argmax(output_cls[sample_idx][:,:seq_len],axis=0)
                
                # Here we prepare to collect similarity scores.
                # Each entry: (frame_idx, true_class, epsilon, direction, sim_same, sim_diff)
                # perturb_results = []
                
                # # Define epsilon values (perturbation magnitudes), using 5 steps from 0 to 0.1.
                # epsilons = torch.linspace(0, 2, steps=100)
                # n_text_classes = action_text_embedding.shape[0]

                # # Ensure joint_text_embedding is on CPU for computation (if necessary)
                # text_emb = action_text_embedding.cpu()
                # # print(cur_feature.shape, text_emb.shape)
                # frame_num = seq_len
                # if perturb_eval:
                #     frame_num = r_b - l_b
                
                # for frame_idx in range(frame_num):
                #     # Get the original feature for the frame and its true class label.
                #     f_orig = cur_feature[frame_idx]
                #     true_class = int(gt_cls[frame_idx])
                #     pred_class = int(pred_cls[frame_idx])
                #     # Same-class text feature
                #     text_same = text_emb[true_class]
                #     # Different-class text features: choose all classes except the true one.
                #     diff_indices = [i for i in range(n_text_classes) if i != true_class]
                #     text_diff = text_emb[diff_indices]

                #     # Compute similarity for the original (no perturbation)
                #     same_sim_orig = torch.abs(F.cosine_similarity(f_orig.unsqueeze(0), text_same.unsqueeze(0))).item()
                #     # Repeat f_orig to match number of different classes and average cosine similarity.
                #     diff_sim_orig = torch.abs(F.cosine_similarity(f_orig.unsqueeze(0).repeat(len(diff_indices), 1), text_diff))
                #     diff_sim_orig_mean = diff_sim_orig.mean().item()
                #     perturb_results.append((frame_idx, pred_class, true_class, 0.0, 'none', same_sim_orig, diff_sim_orig.tolist(), diff_sim_orig_mean))
                    
                #     # For each nonzero epsilon, add noise in both positive and negative directions.
                #     for eps in epsilons[1:]:
                #         # Generate a normalized random noise vector for f_orig.
                #         # noise = torch.randn_like(f_orig)
                        
                #         # noise = noise / (noise.norm() + 1e-8)
                #         for direction in ['positive', 'negative']:
                #             if direction == 'positive':
                #                 noise = torch.normal(eps, 1, size=f_orig.size())
                #                 f_perturbed = f_orig + noise
                #             else:
                #                 noise = torch.normal(eps, 1, size=f_orig.size())
                #                 f_perturbed = f_orig - noise
                #             sim_same = torch.abs(F.cosine_similarity(f_perturbed.unsqueeze(0), text_same.unsqueeze(0))).item()
                #             sim_diff = torch.abs(F.cosine_similarity(f_perturbed.unsqueeze(0).repeat(len(diff_indices), 1), text_diff))
                #             perturb_results.append((frame_idx, pred_class, true_class, eps.item(), direction, sim_same, sim_diff.tolist(), sim_diff.mean().item()))
                # # print(perturb_results)
                # # exit()
                # # Convert perturbation results to a DataFrame for plotting.
                # df = pd.DataFrame(
                #     perturb_results,
                #     columns=['frame', 'pred', 'class', 'epsilon', 'direction', 'sim_same', 'sim_diff', 'sim_diff_mean']
                # )
                # df.to_csv(os.path.join(result_path, f'perturbation_similarity_results{batch_idx:03d}-{sample_idx}_perturb{orig_b}-{pert_b}.csv'), index=False)
                # # ---------- Plot 1: Frame-wise similarity curves per perturbation magnitude ----------
                # for direction in ['none', 'positive', 'negative']:
                #     df_dir = df[df['direction'] == direction]
                #     plt.figure(figsize=(10, 6))
                #     for frm in sorted(df_dir['frame'].unique()):
                #         df_frame = df_dir[df_dir['frame'] == frm]
                #         plt.plot(df_frame['epsilon'], df_frame['sim_same'], marker='o', label=f'Frame {frm} Same')
                #         plt.plot(df_frame['epsilon'], df_frame['sim_diff'], marker='x', label=f'Frame {frm} Diff')
                #     plt.title(f"Frame-wise Cosine Similarity vs Epsilon ({direction} noise)")
                #     plt.xlabel("Perturbation Magnitude (epsilon)")
                #     plt.ylabel("Cosine Similarity")
                #     plt.legend(fontsize=8, loc='best')
                #     plt.tight_layout()
                #     plt.savefig(os.path.join(result_path, f"frame_similarity_{direction}_sample{sample_idx}_batch{batch_idx}.png"))
                #     plt.close()
                
                # # ---------- Plot 2: Class-wise average similarity vs epsilon ----------
                # unique_classes = df['class'].unique()
                # for direction in ['none', 'positive', 'negative']:
                #     df_dir = df[df['direction'] == direction]
                #     plt.figure(figsize=(10, 6))
                #     for cls in unique_classes:
                #         df_cls = df_dir[df_dir['class'] == cls]
                #         # Average similarity per epsilon value.
                #         grouped = df_cls.groupby('epsilon')[['sim_same', 'sim_diff']].mean().reset_index()
                #         plt.plot(grouped['epsilon'], grouped['sim_same'], marker='o', label=f'Class {cls} Same')
                #         plt.plot(grouped['epsilon'], grouped['sim_diff'], marker='x', label=f'Class {cls} Diff')
                #     plt.title(f"Class-wise Average Cosine Similarity vs Epsilon ({direction} noise)")
                #     plt.xlabel("Perturbation Magnitude (epsilon)")
                #     plt.ylabel("Cosine Similarity")
                #     plt.legend(fontsize=8, loc='best')
                #     plt.tight_layout()
                #     plt.savefig(os.path.join(result_path, f"class_similarity_{direction}_sample{sample_idx}_batch{batch_idx}.png"))
                #     plt.close()
                
                # # ---------- Plot 3: Perturbation analysis heatmap (Frame x Epsilon) ----------
                # # Heatmaps for same-class and different-class similarities.
                # for sim_type in ['sim_same', 'sim_diff']:
                #     pivot_table = df.pivot(index='frame', columns='epsilon', values=sim_type)
                #     plt.figure(figsize=(8, 6))
                #     plt.imshow(pivot_table, aspect='auto', origin='lower', cmap='viridis')
                #     plt.colorbar(label="Cosine Similarity")
                #     plt.xlabel("Perturbation Magnitude (epsilon)")
                #     plt.ylabel("Frame Index")
                #     plt.title(f"Heatmap of {sim_type} (aggregated over both directions)")
                #     plt.xticks(ticks=range(len(pivot_table.columns)), labels=[f"{eps:.3f}" for eps in pivot_table.columns])
                #     plt.yticks(ticks=range(len(pivot_table.index)), labels=pivot_table.index)
                #     plt.tight_layout()
                #     plt.savefig(os.path.join(result_path, f"heatmap_{sim_type}_sample{sample_idx}_batch{batch_idx}.png"))
                #     plt.close()

                # End of perturbation analysis for this sample
            for sample_idx in range(x.size(0)):
                if sample_idx > 0:
                    continue

                seq_len = mask_np[sample_idx].sum().astype(int)
                cur_feature = torch.from_numpy(feature_np[sample_idx][:seq_len])
                cur_feature_split = torch.from_numpy(feature_split_np[sample_idx][:seq_len])
                gt_boundary = b_np[sample_idx][:seq_len]
                gt_boundary = gt_boundary[0]
                # print(gt_boundary.shape)
                gt_cls = t_np[sample_idx][:seq_len]
                out_cls = output_cls[sample_idx][:seq_len].transpose(1, 0)
                # print("gt_cls: ", gt_cls.shape)
                # print("out_cls: ", out_cls.shape)
                prob_boundary = torch.from_numpy(output_bound_np[sample_idx][:seq_len])
                curv, curv_reciprocal, movavg = curvature_estimation(cur_feature, 10, 0)
                cross_ = TSConstraintLoss()
                curv_reciprocal = cross_._remap(torch.from_numpy(curv_reciprocal)).to("cpu").data.numpy()
                curv_np.append(curv_reciprocal)
                
                if save_curvature:
                    def normalize_curvature(curve):
                        eps = 1e-8
                        min_val = curve.min()
                        max_val = curve.max()
                        if (max_val - min_val) < eps:
                            return np.zeros_like(curve)
                        return (curve - min_val) / (max_val - min_val + eps)

                    try:
                        vis_dir = os.path.join(result_path, "curvature_prob_wogt")
                        os.makedirs(vis_dir, exist_ok=True)

                        show_gt_boundary = False  # ← 是否显示红色虚线边界

                        # boundary_indices 是 groundtruth 边界点
                        boundary_indices = np.where(gt_boundary > 0.5)[0]

                        # 曲率归一化（可选）
                        # curv_reciprocal = normalize_curvature(curv_reciprocal)

                        plt.figure(figsize=(20, 4))

                        # -------- 子图1: 曲率分数 --------
                        ax1 = plt.subplot(2, 1, 1)
                        ax1.plot(curv_reciprocal, color='blue', linewidth=3.5)
                        if show_gt_boundary:
                            for idx in boundary_indices:
                                if 0 <= idx < len(curv_reciprocal):
                                    ax1.axvline(x=idx, color='red', linestyle='--', linewidth=1, alpha=0.6)

                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        for spine in ax1.spines.values():
                            spine.set_visible(False)

                        # -------- 子图2: 边界分数 --------
                        ax2 = plt.subplot(2, 1, 2)
                        ax2.plot(prob_boundary, color='orange', linewidth=3.5)
                        if show_gt_boundary:
                            for idx in boundary_indices:
                                if 0 <= idx < len(prob_boundary):
                                    ax2.axvline(x=idx, color='red', linestyle='--', linewidth=1, alpha=0.6)

                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        for spine in ax2.spines.values():
                            spine.set_visible(False)

                        # 保存图像
                        plt.tight_layout()
                        save_path = os.path.join(vis_dir, f"batch{batch_idx}_sample{sample_idx}_curvature.png")
                        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print(f"✅ Saved to {save_path}")

                    except Exception as e:
                        print(f"Error generating plot: {str(e)}")
                    # try:
                    #     vis_dir = os.path.join(result_path, "curvature_prob")
                    #     os.makedirs(vis_dir, exist_ok=True)

                    #     plt.figure(figsize=(18, 18))

                    #     def normalize_curvature(curve):
                    #         eps = 1e-8
                    #         min_val = curve.min()
                    #         max_val = curve.max()
                    #         if (max_val - min_val) < eps:
                    #             return torch.zeros_like(curve)
                    #         return (curve - min_val) / (max_val - min_val + eps)
                        
                    #     # action_curvature = normalize_curvature(compute_curvature(cur_feature))
                        
                    #     # loc_curvature = normalize_curvature(compute_curvature(cur_feature_split))

                    #     # Plot Action Feature Curvature
                    #     plt.subplot(4, 1, 1)
                    #     plt.plot(curv_reciprocal, label='Action Curvature', color='blue')
                    #     plt.plot(movavg, label='Moving Average', color='red')
                    #     # print("curv_reciprocal", curv_reciprocal.shape)
                    #     boundary_indices = np.where(gt_boundary > 0.5)[0]
                    #     for idx in boundary_indices:
                    #         if 0 <= idx < len(curv):
                    #             plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6)
                    #     plt.title("Action Feature Curvature with Groundtruth Boundaries")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Curvature")
                    #     plt.legend()

                    #     # Plot Boundary Probability
                    #     plt.subplot(4, 1, 2)
                    #     plt.plot(prob_boundary, label='Boundary Probability', color='orange')
                    #     # print("prob_boundary: ", prob_boundary.shape)
                    #     # Mark groundtruth boundaries
                    #     for idx in boundary_indices:
                    #         if 0 <= idx < len(prob_boundary):
                    #             plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6, label='Boundary')
                    #     plt.title("Boundary Probability with Correct and Incorrect Predictions")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Probability")
                    #     plt.legend()
                    #     plt.subplot(4, 1, 3)
                    #     prob = 1 / (1 + np.exp(-prob_boundary))
                    #     plt.plot(prob, label='Boundary Probability', color='orange')
                    #     # print("prob_boundary: ", prob_boundary.shape)
                    #     # Mark groundtruth boundaries
                    #     for idx in boundary_indices:
                    #         if 0 <= idx < len(prob_boundary):
                    #             plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6, label='Boundary')
                    #     plt.title("sigmoid Boundary Probability with Correct and Incorrect Predictions")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Probability")
                    #     plt.legend()
                    #     plt.subplot(4, 1, 4)
                    #     # Add scatter plot for correct and incorrect predictions
                    #     incorrect_indices = np.where(gt_cls != out_cls.argmax(axis=1))[0]
                    #     # print(incorrect_indices)
                    #     # print(np.where(gt_boundary != (prob_boundary > boundary_th).numpy()))
                    #     # exit()
                    #     for idx in incorrect_indices:
                    #         if 0 <= idx < len(gt_cls):
                    #             plt.scatter(idx, out_cls.argmax(axis=1)[idx], color='red', label=f'Incorrect: GT={gt_cls[idx]}, Pred={out_cls[idx][gt_cls[idx]]:.2f}')
                        
                    #     # Mark correct predictions in green
                    #     correct_indices = np.where(gt_cls == out_cls.argmax(axis=1))[0]
                    #     for idx in correct_indices:
                    #         if 0 <= idx < len(gt_cls):
                    #             plt.scatter(idx, out_cls.argmax(axis=1)[idx], color='green', label=f'Correct: {out_cls[idx].max():.2f}')
                    #     plt.title("cls Probability with Correct and Incorrect Predictions")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Class")
                    #     plt.legend()
                        
                        
                        
                    #     # Save plot
                    #     # plt.tight_layout()
                    #     plt.savefig(os.path.join(vis_dir, f"batch{batch_idx}_sample{sample_idx}_curvature.png"))
                    #     plt.close()
                    # except Exception as e:
                    #     print(f"Error generating plot: {str(e)}")

            curv_np = np.array(curv_np)
            # Original processing logic
            refined_output_cls = postprocessor(output_cls, boundaries=output_bound, masks=mask_np)
            scores_before_refinement.update(output_cls, t_np)
            scores_bound.update(output_bound, b_np, mask_np)
            scores_after_refinement.update(refined_output_cls, t_np)
            IOU_before_refinement.update(output_cls, t_np, output_bound, curv_np)
            output_bound = 1 / (1 + np.exp(-output_bound))
            IOU_after_refinement.update(refined_output_cls, t_np, output_bound, curv_np)

            
    with open(os.path.join(result_path, "evaluation_config.txt"), "w") as f:
        f.write(str(config))       
    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())
    print("IOU before refinement:", IOU_before_refinement.get_scores())
    # IOU_before_refinement.plot_iou(os.path.join(result_path+'/before', "iou_before_refinement.png"))
    # IOU_after_refinement.plot_iou(os.path.join(result_path+'/after', "iou_after_refinement.png"))
    # Save results
    scores_before_refinement.save_scores(os.path.join(result_path, "test_as_before_refine.csv"))
    scores_before_refinement.save_confusion_matrix(os.path.join(result_path, "test_c_matrix_before_refinement.csv"))
    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))
    scores_after_refinement.save_scores(os.path.join(result_path, "test_as_after_majority_vote.csv"))
    scores_after_refinement.save_confusion_matrix(os.path.join(result_path, "test_c_matrix_after_majority_vote.csv"))
    
def evaluate_dxr(
    val_loader: DataLoader,
    model: nn.Module,
    model_text,
    joint_text_list,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str = None,
    config: str = None,
    refinement_method: Optional[str] = None,
    save_curvature: bool = False,
    save_tsne: bool = False,
    multi_test: bool = False,
) -> None:
    if save_curvature:
        plt.switch_backend('Agg')  # Use Agg backend for headless environments
    id2class_map = get_id2class_map(dataset, dataset_dir=dataset_dir)
    # print("id2class_map: ", id2class_map)
    # exit()
    postprocessor = PostProcessor(refinement_method, boundary_th)
    scores_before_refinement = ScoreMeter(
        id2class_map=id2class_map,
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )
    scores_after_refinement = ScoreMeter(
        id2class_map=id2class_map,
        iou_thresholds=iou_thresholds,
    )
    IOU_before_refinement = IOUMeter(
        id2class_map=id2class_map,
        max_len=9000,
    )

    model.eval()
    test = 2
    if save_tsne:
        save_tsne_path = os.path.join(result_path, "tsne")
        if not os.path.exists(save_tsne_path):
            os.makedirs(save_tsne_path)
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(val_loader)):
            # if batch_idx == test:
            #     break
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)
            joint_text_list = joint_text_list.to(device)

            joint_text_embedding = model_text(joint_text_list).float()
            if multi_test:
                out_cls_ori, out_bound_ori, output_feature, bound_feature = model(x, mask, joint_text_embedding)
                # === 上采样输入（T方向放大2倍）===
                x_up = F.interpolate(x, scale_factor=(1, 2), mode="bilinear", align_corners=False)  # V 不变, T×2
                mask_up = F.interpolate(mask.unsqueeze(1).float(), scale_factor=(1, 2), mode="nearest").squeeze(1).bool()
                out_cls_up, out_bound_up, _, _ = model(x_up, mask_up, joint_text_embedding)

                # 下采样输入（T方向缩小为 1/2）
                x_down = F.interpolate(x, scale_factor=(1, 0.5), mode="bilinear", align_corners=False)
                mask_down = F.interpolate(mask.unsqueeze(1).float(), scale_factor=(1, 0.5), mode="nearest").squeeze(1).bool()
                out_cls_down, out_bound_down, _, _ = model(x_down, mask_down, joint_text_embedding)

                # === 插值回原始 T 长度 ===
                T_orig = x.shape[-1]
                out_cls_up = F.interpolate(out_cls_up, size=T_orig, mode="linear", align_corners=False)
                out_cls_down = F.interpolate(out_cls_down, size=T_orig, mode="linear", align_corners=False)

                out_bound_up = F.interpolate(out_bound_up, size=T_orig, mode="linear", align_corners=False)
                out_bound_down = F.interpolate(out_bound_down, size=T_orig, mode="linear", align_corners=False)

                # === 融合（平均）===
                output_cls = (out_cls_ori + out_cls_up + out_cls_down) / 3
                output_bound = (out_bound_ori + out_bound_up + out_bound_down) / 3
            else:
                output_cls, output_bound, output_feature, bound_feature = model(x, mask, joint_text_embedding)

            

            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()
            x_np = x.to("cpu").data.numpy()
            t_np = t.to("cpu").data.numpy()
            b_np = b.to("cpu").data.numpy()
            mask_np = mask.to("cpu").data.numpy()
            feature_np = output_feature.to("cpu").data.numpy().transpose(0, 2, 1)
            feature_split_np = bound_feature.to("cpu").data.numpy().transpose(0, 2, 1)
            output_bound_np = output_bound.transpose(0, 2, 1)
            curv_np = []

            # print(output_feature.shape)
            # print(t.shape)
            if save_tsne:
                N, C, T = output_feature.shape
                output_feature_circle = output_feature.transpose(1, 2).reshape(N*T, C)
                t_circle = t.reshape(N*T, )
                plot_tsne(output_feature_circle, t_circle, os.path.join(save_tsne_path, f"tsne_action_{batch_idx}.png"))

            for sample_idx in range(x.size(0)):
                if sample_idx > 0:
                    continue

                seq_len = mask_np[sample_idx].sum().astype(int)
                cur_feature = torch.from_numpy(feature_np[sample_idx][:seq_len])
                cur_feature_split = torch.from_numpy(feature_split_np[sample_idx][:seq_len])
                gt_boundary = b_np[sample_idx][:seq_len]
                gt_boundary = gt_boundary[0]
                # print(gt_boundary.shape)
                gt_cls = t_np[sample_idx][:seq_len]
                out_cls = output_cls[sample_idx][:seq_len].transpose(1, 0)
                # print("gt_cls: ", gt_cls.shape)
                # print("out_cls: ", out_cls.shape)
                prob_boundary = torch.from_numpy(output_bound_np[sample_idx][:seq_len])
                curv, curv_reciprocal, movavg = curvature_estimation(cur_feature, 10, 0)
                # curv_reciprocal = 
                curv_np.append(curv_reciprocal)
                if save_curvature:
                    # try:
                    #     vis_dir = os.path.join(result_path, "curvature_prob")
                    #     os.makedirs(vis_dir, exist_ok=True)

                    #     plt.figure(figsize=(18, 18))

                    #     def normalize_curvature(curve):
                    #         eps = 1e-8
                    #         min_val = curve.min()
                    #         max_val = curve.max()
                    #         if (max_val - min_val) < eps:
                    #             return torch.zeros_like(curve)
                    #         return (curve - min_val) / (max_val - min_val + eps)
                        
                    #     # action_curvature = normalize_curvature(compute_curvature(cur_feature))
                        
                    #     # loc_curvature = normalize_curvature(compute_curvature(cur_feature_split))

                    #     # Plot Action Feature Curvature
                    #     plt.subplot(3, 1, 1)
                    #     plt.plot(curv_reciprocal, label='Action Curvature', color='blue')
                    #     plt.plot(movavg, label='Moving Average', color='red')
                    #     # print("curv_reciprocal", curv_reciprocal.shape)
                    #     boundary_indices = np.where(gt_boundary > 0.5)[0]
                    #     for idx in boundary_indices:
                    #         if 0 <= idx < len(curv):
                    #             plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6)
                    #     plt.title("Action Feature Curvature with Groundtruth Boundaries")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Curvature")
                    #     plt.legend()

                    #     # Plot Boundary Probability
                    #     plt.subplot(3, 1, 2)
                    #     plt.plot(prob_boundary, label='Boundary Probability', color='orange')
                    #     # print("prob_boundary: ", prob_boundary.shape)
                    #     # Mark groundtruth boundaries
                    #     for idx in boundary_indices:
                    #         if 0 <= idx < len(prob_boundary):
                    #             plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6, label='Boundary')
                    #     plt.title("Boundary Probability with Correct and Incorrect Predictions")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Probability")
                    #     plt.legend()
                    #     plt.subplot(3, 1, 3)
                    #     # Add scatter plot for correct and incorrect predictions
                    #     incorrect_indices = np.where(gt_cls != out_cls.argmax(axis=1))[0]
                    #     # print(incorrect_indices)
                    #     # print(np.where(gt_boundary != (prob_boundary > boundary_th).numpy()))
                    #     # exit()
                    #     for idx in incorrect_indices:
                    #         if 0 <= idx < len(gt_cls):
                    #             plt.scatter(idx, out_cls.argmax(axis=1)[idx], color='red', label=f'Incorrect: GT={gt_cls[idx]}, Pred={out_cls[idx][gt_cls[idx]]:.2f}')
                        
                    #     # Mark correct predictions in green
                    #     correct_indices = np.where(gt_cls == out_cls.argmax(axis=1))[0]
                    #     for idx in correct_indices:
                    #         if 0 <= idx < len(gt_cls):
                    #             plt.scatter(idx, out_cls.argmax(axis=1)[idx], color='green', label=f'Correct: {out_cls[idx].max():.2f}')
                    #     plt.title("cls Probability with Correct and Incorrect Predictions")
                    #     plt.xlabel("Frame Index")
                    #     plt.ylabel("Class")
                    #     plt.legend()
                        

                    #     # Save plot
                    #     # plt.tight_layout()
                    #     plt.savefig(os.path.join(vis_dir, f"batch{batch_idx}_sample{sample_idx}_curvature.png"))
                    #     plt.close()
                    # except Exception as e:
                    #     print(f"Error generating plot: {str(e)}")
                    def normalize_curvature(curve):
                        eps = 1e-8
                        min_val = curve.min()
                        max_val = curve.max()
                        if (max_val - min_val) < eps:
                            return np.zeros_like(curve)
                        return (curve - min_val) / (max_val - min_val + eps)

                    try:
                        vis_dir = os.path.join(result_path, "curvature_prob")
                        os.makedirs(vis_dir, exist_ok=True)

                        show_gt_boundary = True  # ← 是否显示红色虚线边界

                        # boundary_indices 是 groundtruth 边界点
                        boundary_indices = np.where(gt_boundary > 0.5)[0]

                        # 曲率归一化（可选）
                        # curv_reciprocal = normalize_curvature(curv_reciprocal)

                        plt.figure(figsize=(12, 4))

                        # -------- 子图1: 曲率分数 --------
                        ax1 = plt.subplot(2, 1, 1)
                        ax1.plot(curv_reciprocal, color='blue', linewidth=1.5)
                        if show_gt_boundary:
                            for idx in boundary_indices:
                                if 0 <= idx < len(curv_reciprocal):
                                    ax1.axvline(x=idx, color='red', linestyle='--', linewidth=1, alpha=0.6)

                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        for spine in ax1.spines.values():
                            spine.set_visible(False)

                        # -------- 子图2: 边界分数 --------
                        ax2 = plt.subplot(2, 1, 2)
                        ax2.plot(prob_boundary, color='orange', linewidth=1.5)
                        if show_gt_boundary:
                            for idx in boundary_indices:
                                if 0 <= idx < len(prob_boundary):
                                    ax2.axvline(x=idx, color='red', linestyle='--', linewidth=1, alpha=0.6)

                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        for spine in ax2.spines.values():
                            spine.set_visible(False)

                        # 保存图像
                        plt.tight_layout()
                        save_path = os.path.join(vis_dir, f"batch{batch_idx}_sample{sample_idx}_curvature.png")
                        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print(f"✅ Saved to {save_path}")

                    except Exception as e:
                        print(f"Error generating plot: {str(e)}")

            curv_np = np.array(curv_np)
            # Original processing logic
            refined_output_cls = postprocessor(output_cls, boundaries=output_bound, masks=mask_np)
            scores_before_refinement.update(output_cls, t_np)
            scores_bound.update(output_bound, b_np, mask_np)
            scores_after_refinement.update(refined_output_cls, t_np)
            IOU_before_refinement.update(output_cls, t_np, output_bound, curv_np)
    # with open(os.path.join(result_path, "evaluation_config.txt"), "w") as f:
    #     f.write(str(config))       
    # print("Before refinement:", scores_before_refinement.get_scores())
    # print("Boundary scores:", scores_bound.get_scores())
    # print("After refinement:", scores_after_refinement.get_scores())
    scores_before = scores_before_refinement.get_scores()
    Scores = scores_after_refinement.get_scores()
    IOUs =  IOU_before_refinement.get_scores()
    # IOU_before_refinement.plot_iou(os.path.join(result_path, "iou_before_refinement.png"))
    # Save results
    # scores_before_refinement.save_scores(os.path.join(result_path, "test_as_before_refine.csv"))
    # scores_before_refinement.save_confusion_matrix(os.path.join(result_path, "test_c_matrix_before_refinement.csv"))
    # scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))
    # scores_after_refinement.save_scores(os.path.join(result_path, "test_as_after_majority_vote.csv"))
    # scores_after_refinement.save_confusion_matrix(os.path.join(result_path, "test_c_matrix_after_majority_vote.csv"))
    return Scores[0], Scores[1], Scores[2][0], Scores[2][1], Scores[2][2], IOUs[0], scores_before
