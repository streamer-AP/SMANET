import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint

from misc.layer import Gaussianlayer, PointsToHeatmap, WeightedBCELoss, FocalTverskyLoss
from model.VGG.VGG16_FPN import VGG16_FPN_Encoder
from model.ViT.models_crossvit import CrossAttentionBlock, FeatureFusionModule
from model.decoder import GlobalDecoder, InOutDecoder, ShareDecoder
from model.points_from_den import *
from model.probability_decoder.probability_decoder import ProbabilityDecoder


def remap_legacy_state_dict(state_dict):
    replacements = (
        ("Extractor.", "image_encoder."),
        ("Gaussian.", "gaussian_smoother."),
        ("unet_decoder.", "probability_decoder."),
    )
    remapped = {}
    for key, value in state_dict.items():
        # Older checkpoints may contain a stale top-level alpha parameter;
        # current SMANet gates with the predicted probability map directly.
        if key in {"alpha", "module.alpha"}:
            continue
        new_key = key
        for old_prefix, new_prefix in replacements:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
            module_prefix = "module." + old_prefix
            if new_key.startswith(module_prefix):
                new_key = "module." + new_prefix + new_key[len(module_prefix):]
                break
        remapped[new_key] = value
    return remapped


class SMANet(nn.Module):
    def __init__(self, cfg, cfg_data):
        super().__init__()
        self.cfg = cfg
        self.cfg_data = cfg_data
        self.flow_size_printed = False
        if cfg.encoder != 'VGG16_FPN':
            raise ValueError('Only VGG16_FPN is supported in this release.')
        self.image_encoder = VGG16_FPN_Encoder()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.share_cross_attention = nn.ModuleList([nn.ModuleList([
            CrossAttentionBlock(cfg.cross_attn_embed_dim, cfg.cross_attn_num_heads, cfg.mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(cfg.cross_attn_depth)])
            for _ in range(3)])

        self.share_cross_attention_norm = norm_layer(cfg.cross_attn_embed_dim)

        self.feature_fuse = FeatureFusionModule(self.cfg.FEATURE_DIM)
        self.global_decoder = GlobalDecoder()
        self.share_decoder = ShareDecoder()
        self.in_out_decoder = InOutDecoder()
        self.criterion = torch.nn.MSELoss()

        self.flow_embed = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.gaussian_smoother = Gaussianlayer(
            sigma=[cfg.DENSITY_SIGMA],
            kernel_size=cfg.DENSITY_KERNEL_SIZE,
        )
        self.points_to_heatmap = PointsToHeatmap(
            sigma=cfg.PROBABILITY_SIGMA,
            kernel_size=cfg.PROBABILITY_KERNEL_SIZE,
        )
        self.weighted_bce = WeightedBCELoss(pos_weight=cfg.PROBABILITY_POS_WEIGHT)
        self.focal_tversky = FocalTverskyLoss(
            alpha=cfg.FOCAL_TVERSKY_ALPHA,
            beta=cfg.FOCAL_TVERSKY_BETA,
            gamma=cfg.FOCAL_TVERSKY_GAMMA,
            smooth=cfg.FOCAL_TVERSKY_SMOOTH,
        )
        self.probability_decoder = ProbabilityDecoder(n_channels=288, n_classes=1)

    def _se_fusion(self, img_feat, flow_feat):
        return torch.cat([img_feat, flow_feat], dim=1)

    def forward(self, img, target, flows, epoch):

        forward_flows = flows[0].to(img.device)
        backward_flows = flows[1].to(img.device)

        if not self.flow_size_printed:
            print(f"\n[Flow] Loaded optical flow from disk, forward shape: {forward_flows.shape}\n")
            self.flow_size_printed = True

        features, f_list = checkpoint(self.image_encoder, img, use_reentrant=False)

        B, C, H, W = features[-1].shape
        self.feature_scale = H / img.shape[2]

        if epoch <= self.cfg.FEATURE_FREEZE_EPOCHS:
            features_for_probability = [f.detach() for f in features]
            f_list_for_probability = [f.detach() for f in f_list]
        else:
            features_for_probability = features
            f_list_for_probability = f_list

        def normalize_flow(flow_tensor):
            normalized_flows = []
            for flow in flow_tensor:
                max_val = torch.max(torch.abs(flow))
                if max_val > 0:
                    normalized_flow = flow / (max_val + 1e-6)
                else:
                    normalized_flow = flow
                normalized_flows.append(normalized_flow)
            return torch.stack(normalized_flows)

        forward_flows_normalized = normalize_flow(forward_flows)
        backward_flows_normalized = normalize_flow(backward_flows)

        target_h, target_w = H, W
        forward_flows_down = F.interpolate(forward_flows_normalized, size=(target_h, target_w), mode='bilinear', align_corners=False)
        backward_flows_down = F.interpolate(backward_flows_normalized, size=(target_h, target_w), mode='bilinear', align_corners=False)

        flow_feat_out = checkpoint(self.flow_embed, forward_flows_down, use_reentrant=False)
        flow_feat_in = checkpoint(self.flow_embed, backward_flows_down, use_reentrant=False)

        img_feat_1 = features_for_probability[-1][0::2]
        img_feat_2 = features_for_probability[-1][1::2]

        decoder_input_out = self._se_fusion(img_feat_1, flow_feat_out)
        decoder_input_in = self._se_fusion(img_feat_2, flow_feat_in)

        skips_img1 = [f_list_for_probability[i][0::2] for i in range(len(f_list_for_probability))]
        skips_img2 = [f_list_for_probability[i][1::2] for i in range(len(f_list_for_probability))]

        outflow_probability_logits = checkpoint(self.probability_decoder, decoder_input_out, skips_img1, use_reentrant=False)
        inflow_probability_logits = checkpoint(self.probability_decoder, decoder_input_in, skips_img2, use_reentrant=False)

        pre_global_den = checkpoint(self.global_decoder, features[-1], use_reentrant=False)
        all_loss = {}
        gt_in_out_dot_map = torch.zeros_like(pre_global_den)
        gt_share_dot_map = torch.zeros_like(pre_global_den)
        share_features = None
        img_pair_num = img.size(0) // 2

        for l_num in range(len(self.share_cross_attention)):
            _, _, H_feat, W_feat = features[l_num].shape
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])

            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = checkpoint(self.share_cross_attention[l_num][i], q1, kv, H_feat, W_feat, use_reentrant=False)

                q1 = self.share_cross_attention_norm(q1)

                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = checkpoint(self.share_cross_attention[l_num][i], q2, kv, H_feat, W_feat, use_reentrant=False)

                q2 = self.share_cross_attention_norm(q2)

                share_results.append(q1)
                share_results.append(q2)

            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        for pair_idx in range(img_pair_num):
            points0 = target[pair_idx * 2]['points']
            points1 = target[pair_idx * 2 + 1]['points']

            share_mask0 = target[pair_idx * 2]['share_mask0']
            outflow_mask = target[pair_idx * 2]['outflow_mask']
            share_mask1 = target[pair_idx * 2 + 1]['share_mask1']
            inflow_mask = target[pair_idx * 2 + 1]['inflow_mask']

            share_coords0 = points0[share_mask0].long()
            share_coords1 = points1[share_mask1].long()

            gt_share_dot_map[pair_idx * 2, 0, share_coords0[:, 1], share_coords0[:, 0]] = 1
            gt_share_dot_map[pair_idx * 2 + 1, 0, share_coords1[:, 1], share_coords1[:, 0]] = 1

            outflow_coords = points0[outflow_mask].long()
            inflow_coords = points1[inflow_mask].long()

            gt_in_out_dot_map[pair_idx * 2, 0, outflow_coords[:, 1], outflow_coords[:, 0]] = 1
            gt_in_out_dot_map[pair_idx * 2 + 1, 0, inflow_coords[:, 1], inflow_coords[:, 0]] = 1

        pre_share_den = checkpoint(self.share_decoder, share_features, use_reentrant=False)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = checkpoint(self.in_out_decoder, mid_pre_in_out_den, use_reentrant=False)

        gt_global_dot_map = torch.zeros_like(pre_global_den)
        for i, data in enumerate(target):
            points = data['points'].long()
            gt_global_dot_map[i, 0, points[:, 1], points[:, 0]] = 1
        gt_global_den = self.gaussian_smoother(gt_global_dot_map)

        assert pre_global_den.size() == gt_global_den.size()
        global_mse_loss = self.criterion(pre_global_den, gt_global_den * self.cfg_data.DEN_FACTOR)
        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['global'] = global_mse_loss

        gt_share_den = self.gaussian_smoother(gt_share_dot_map)
        assert pre_share_den.size() == gt_share_den.size()
        share_mse_loss = self.criterion(pre_share_den, gt_share_den * self.cfg_data.DEN_FACTOR)
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['share'] = share_mse_loss

        gt_in_out_den = self.gaussian_smoother(gt_in_out_dot_map)

        _, _, h_probability, w_probability = outflow_probability_logits.shape
        _, _, H_img, W_img = img.shape

        outflow_points_list = []
        for i in range(img_pair_num):
            points = target[i * 2]['points'][target[i * 2]['outflow_mask']]
            scaled_points = points.clone()
            if scaled_points.numel() > 0:
                scaled_points[:, 0] = scaled_points[:, 0] * w_probability / W_img
                scaled_points[:, 1] = scaled_points[:, 1] * h_probability / H_img
            outflow_points_list.append(scaled_points)

        inflow_points_list = []
        for i in range(img_pair_num):
            points = target[i * 2 + 1]['points'][target[i * 2 + 1]['inflow_mask']]
            scaled_points = points.clone()
            if scaled_points.numel() > 0:
                scaled_points[:, 0] = scaled_points[:, 0] * w_probability / W_img
                scaled_points[:, 1] = scaled_points[:, 1] * h_probability / H_img
            inflow_points_list.append(scaled_points)

        gt_heatmap_out = self.points_to_heatmap(img_pair_num, 1, h_probability, w_probability, outflow_points_list)
        gt_heatmap_in = self.points_to_heatmap(img_pair_num, 1, h_probability, w_probability, inflow_points_list)

        if epoch <= self.cfg.PROBABILITY_BCE_EPOCHS:
            loss_out = self.weighted_bce(outflow_probability_logits, gt_heatmap_out)
            loss_in = self.weighted_bce(inflow_probability_logits, gt_heatmap_in)
        else:
            loss_out_wBCE = self.weighted_bce(outflow_probability_logits, gt_heatmap_out)
            loss_out_focal = self.focal_tversky(outflow_probability_logits, gt_heatmap_out)
            loss_in_wBCE = self.weighted_bce(inflow_probability_logits, gt_heatmap_in)
            loss_in_focal = self.focal_tversky(inflow_probability_logits, gt_heatmap_in)

            loss_out = (
                self.cfg.PROBABILITY_BCE_WEIGHT * loss_out_wBCE
                + self.cfg.PROBABILITY_FOCAL_WEIGHT * loss_out_focal
            )
            loss_in = (
                self.cfg.PROBABILITY_BCE_WEIGHT * loss_in_wBCE
                + self.cfg.PROBABILITY_FOCAL_WEIGHT * loss_in_focal
            )
        all_loss['probability_outflow'] = loss_out
        all_loss['probability_inflow'] = loss_in

        outflow_probability = torch.sigmoid(outflow_probability_logits)
        inflow_probability = torch.sigmoid(inflow_probability_logits)

        if outflow_probability.shape[2:] != pre_in_out_den.shape[2:]:
            prob_out_resized = F.interpolate(outflow_probability, size=pre_in_out_den.shape[2:],
                                            mode='bilinear', align_corners=False)
            prob_in_resized = F.interpolate(inflow_probability, size=pre_in_out_den.shape[2:],
                                           mode='bilinear', align_corners=False)
        else:
            prob_out_resized = outflow_probability
            prob_in_resized = inflow_probability

        prob_map = torch.zeros_like(pre_in_out_den)
        prob_map[0::2] = prob_out_resized
        prob_map[1::2] = prob_in_resized

        assert prob_map.size() == pre_in_out_den.size()

        pre_in_out_den = pre_in_out_den * prob_map

        assert prob_map.size() == pre_in_out_den.size(),\
            f"Probability map size {prob_map.size()} doesn't match density map size {pre_in_out_den.size()}"

        assert pre_in_out_den.size() == gt_in_out_den.size()
        in_out_mse_loss = self.criterion(pre_in_out_den, gt_in_out_den * self.cfg_data.DEN_FACTOR)
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['in_out'] = in_out_mse_loss

        return pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss

    def test_forward(self, img, flows):

        forward_flows = flows[0].to(img.device)
        backward_flows = flows[1].to(img.device)

        features, f_list = self.image_encoder(img)
        B, C, H, W = features[-1].shape

        def normalize_flow(flow_tensor):
            normalized_flows = []
            for flow in flow_tensor:
                max_val = torch.max(torch.abs(flow))
                if max_val > 0:
                    normalized_flow = flow / (max_val + 1e-6)
                else:
                    normalized_flow = flow
                normalized_flows.append(normalized_flow)
            return torch.stack(normalized_flows)

        forward_flows_normalized = normalize_flow(forward_flows)
        backward_flows_normalized = normalize_flow(backward_flows)

        target_h, target_w = H, W
        forward_flows_down = F.interpolate(forward_flows_normalized, size=(target_h, target_w), mode='bilinear', align_corners=False)
        backward_flows_down = F.interpolate(backward_flows_normalized, size=(target_h, target_w), mode='bilinear', align_corners=False)

        flow_feat_out = self.flow_embed(forward_flows_down)
        flow_feat_in = self.flow_embed(backward_flows_down)

        img_feat_1 = features[-1][0::2]
        img_feat_2 = features[-1][1::2]

        decoder_input_out = self._se_fusion(img_feat_1, flow_feat_out)
        decoder_input_in = self._se_fusion(img_feat_2, flow_feat_in)

        skips_img1 = [f[0::2] for f in f_list]
        skips_img2 = [f[1::2] for f in f_list]

        outflow_probability_logits = self.probability_decoder(decoder_input_out, skips_img1)
        inflow_probability_logits = self.probability_decoder(decoder_input_in, skips_img2)

        pre_global_den = self.global_decoder(features[-1])
        img_pair_num = img.size(0) // 2
        share_features = None
        for l_num in range(len(self.share_cross_attention)):
            _, _, H_feat, W_feat = features[l_num].shape
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])

            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv , H=H_feat, W=W_feat)
                q1 = self.share_cross_attention_norm(q1)

                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                kv = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv , H=H_feat, W=W_feat)
                q2 = self.share_cross_attention_norm(q2)

                share_results.append(q1)
                share_results.append(q2)

            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        pre_share_den = self.share_decoder(share_features)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den)

        outflow_probability = torch.sigmoid(outflow_probability_logits)
        inflow_probability = torch.sigmoid(inflow_probability_logits)

        if outflow_probability.shape[2:] != pre_in_out_den.shape[2:]:
            prob_out_resized = F.interpolate(outflow_probability, size=pre_in_out_den.shape[2:], mode='bilinear', align_corners=False)
            prob_in_resized = F.interpolate(inflow_probability, size=pre_in_out_den.shape[2:], mode='bilinear', align_corners=False)
        else:
            prob_out_resized = outflow_probability
            prob_in_resized = inflow_probability

        prob_map = torch.zeros_like(pre_in_out_den)
        prob_map[0::2] = prob_out_resized
        prob_map[1::2] = prob_in_resized

        pre_in_out_den = pre_in_out_den * prob_map

        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR

        return pre_global_den, pre_share_den, pre_in_out_den
