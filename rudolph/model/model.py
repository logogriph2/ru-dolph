# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from einops import rearrange

from .utils import init_method_normal
from .transformer import SparseTransformer


class ruDolphModel(torch.nn.Module):
    def __init__(self,
                 device,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 l_text_seq_length=64,
                 r_text_seq_length=64,
                 kernel_size=7,
                 last_kernel_size=9,
                 image_tokens_per_dim=16,
                 image_vocab_size=8192,
                 text_special_tokens=0,
                 image_special_tokens=0,
                 cogview_sandwich_layernorm=True,
                 cogview_pb_relax=True,
                 is_bool_mask=True,
                 mlp_activation='gelu_jit',
                 gradient_checkpointing=None):
        super(ruDolphModel, self).__init__()
        print('Creating object detection mod model')
        self.device = device
        self.image_tokens_per_dim = image_tokens_per_dim
        self.image_seq_length = image_tokens_per_dim ** 2
        self.l_text_seq_length = l_text_seq_length
        self.r_text_seq_length = r_text_seq_length
        self.total_seq_length = self.l_text_seq_length + self.image_seq_length + self.r_text_seq_length
        self.text_special_tokens = text_special_tokens
        self.image_special_tokens = image_special_tokens
        vocab_size = vocab_size + text_special_tokens
        image_vocab_size = image_vocab_size + image_special_tokens
        self.total_vocab_size = vocab_size + image_vocab_size
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        init_method = init_method_normal(std=0.02)

        self.text_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.image_embeddings = torch.nn.Embedding(image_vocab_size, hidden_size)

        # Position embedding (serial).
        self.l_text_pos_embeddings = torch.nn.Embedding(l_text_seq_length + 1, hidden_size)
        self.r_text_pos_embeddings = torch.nn.Embedding(r_text_seq_length + 1, hidden_size)
        self.image_row_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        self.image_col_embeddings = torch.nn.Embedding(image_tokens_per_dim, hidden_size)
        init_method(self.l_text_pos_embeddings.weight)
        init_method(self.r_text_pos_embeddings.weight)
        init_method(self.image_row_embeddings.weight)
        init_method(self.image_col_embeddings.weight)

        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, self.total_vocab_size),
        )

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = SparseTransformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            l_text_seq_length=l_text_seq_length,
            r_text_seq_length=r_text_seq_length,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            image_tokens_per_dim=image_tokens_per_dim,
            cogview_sandwich_layernorm=cogview_sandwich_layernorm,
            cogview_pb_relax=cogview_pb_relax,
            mlp_activation=mlp_activation,
            is_bool_mask=is_bool_mask,
        )

        self.loc_loss_matrix = 2 * torch.ones((self.image_tokens_per_dim, self.image_tokens_per_dim))
        d_i = [k for k in range(self.image_tokens_per_dim)]
        for dx in d_i:
            for dy in d_i:
                self.loc_loss_matrix[dx, dy] = 0.01 * abs(dx - dy) / self.image_tokens_per_dim
        self.loc_loss_matrix = self.loc_loss_matrix.to(self.device)

    def get_param(self, item):
        return getattr(self, item)

    def get_image_pos_embeddings(self, image_input_ids, device, past_length=0):
        input_shape = image_input_ids.size()
        row_ids = torch.div(
            torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device),
            self.image_tokens_per_dim,
            rounding_mode='trunc'
        )
        row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
        col_ids = torch.arange(past_length, input_shape[-1] + past_length,
                               dtype=torch.long, device=device) % self.image_tokens_per_dim
        col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.image_row_embeddings(row_ids) + self.image_col_embeddings(col_ids)

    def forward(
            self,
            input_ids,
            attention_mask,
            allowed_tokens=None,
            return_loss=False,
            use_cache=False,
            cache=None,
            lt_loss_weight=1,
            img_loss_weight=7,
            rt_loss_weight=1,

            category_weight=0,
            loc_loss_weight=0,
            nms_loss_weight=0,
            categories_num=0,
            iou_threshold=0,
            od_seq_len=5,
            bin_size=8,
            r_text_shift=2,
            fake_category_token=None,
            return_hidden_states=False,
            zero_loc_token=None,
            conf_idx=None,
            conf_loss_weight=None,
            iou_bin_tokens=None,
            cat_idx=None,
            loc_len=4,
            moc_iou_token=None
    ):
        device = input_ids.device
        l_text = input_ids[:, :self.l_text_seq_length]
        l_text_range = torch.arange(l_text.shape[1])
        l_text_range += (self.vocab_size - self.l_text_seq_length)
        l_text_range = l_text_range.to(device)
        l_text = torch.where(l_text == 0, l_text_range, l_text)
        l_text_pos = self.l_text_pos_embeddings(torch.arange(l_text.shape[1], device=device))
        l_text_embeddings = self.text_embeddings(l_text) + l_text_pos

        use_image = input_ids.shape[1] > self.l_text_seq_length
        use_r_text = input_ids.shape[1] > self.l_text_seq_length + self.image_seq_length

        embeddings = [l_text_embeddings]
        if use_image:
            image_input_ids = input_ids[:, self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length]
            img_pos = self.get_image_pos_embeddings(image_input_ids, past_length=0, device=device)
            image_embeddings = self.image_embeddings(image_input_ids) + img_pos
            embeddings.append(image_embeddings)

        if use_r_text:
            r_text = input_ids[:, self.l_text_seq_length + self.image_seq_length:]
            r_text_pos = self.r_text_pos_embeddings(torch.arange(r_text.shape[1], device=device))
            r_text_embeddings = self.text_embeddings(r_text) + r_text_pos
            embeddings.append(r_text_embeddings)

        embeddings = torch.cat(embeddings, dim=1)

        alpha = 0.1
        embeddings = embeddings * alpha + embeddings.detach() * (1 - alpha)

        attention_mask = attention_mask[:, :, :embeddings.shape[1], :embeddings.shape[1]]
        transformer_output, present_cache, hidden_states = self.transformer(
            embeddings, attention_mask, cache=cache, use_cache=use_cache,
            gradient_checkpointing=self.gradient_checkpointing
        )

        logits = self.to_logits(transformer_output)

        if return_loss is False:
            outputs = (logits, present_cache)
            if return_hidden_states:
                outputs += (hidden_states,)
            return outputs

        logits = rearrange(logits, 'b n c -> b c n')
        l_text_logits = logits[
                        :, :self.vocab_size, :self.l_text_seq_length if use_image else self.l_text_seq_length - 1
                        ].contiguous().float()
        labels = [l_text[:, 1:]]

        if use_image:
            labels.append(image_input_ids)
            a, b = self.l_text_seq_length, self.l_text_seq_length + self.image_seq_length - 1
            image_logits = logits[:, self.vocab_size:, a:b].contiguous().float()
        if use_r_text:
            if allowed_tokens:
                r_text_logits = logits[:, allowed_tokens, -self.r_text_seq_length:-1].contiguous().float()
            else:
                r_text_logits = logits[:, :self.vocab_size, -self.r_text_seq_length:-1].contiguous().float()
            labels.append(r_text)
        labels = torch.cat(labels, dim=1).contiguous().long()

        loss, loss_weights, loss_values = 0, 0, {}
        loss_l_text = F.cross_entropy(
            l_text_logits,
            labels[:, :self.l_text_seq_length]
        )
        loss_values['l_text_loss'] = loss_l_text.data.detach().float()

        if lt_loss_weight:
            loss += loss_l_text * lt_loss_weight
            loss_weights += lt_loss_weight

        if use_image:
            loss_img = F.cross_entropy(
                image_logits,
                labels[:, self.l_text_seq_length:self.l_text_seq_length + self.image_seq_length - 1]
            )
            loss_values['image_loss'] = loss_img.data.detach().float()
            if img_loss_weight:
                loss += loss_img * img_loss_weight
                loss_weights += img_loss_weight

        if use_r_text:
            if fake_category_token is not None and torch.where(labels == fake_category_token)[1].size()[0] > 0:
                fake_idxs = torch.where(labels == fake_category_token)[1]
                fake_start = min(fake_idxs) - od_seq_len + 1

                label_idxs = []
                for k in range((fake_start - self.l_text_seq_length - self.image_seq_length) // od_seq_len):
                    obj_start = self.l_text_seq_length + self.image_seq_length + 1 + k * od_seq_len
                    obj_seq = list([obj_start + li for li in range(loc_len)])
                    obj_seq += [obj_start + cat_idx]
                    if conf_idx is not None and moc_iou_token is not None:
                        obj_seq += [obj_start + conf_idx]
                    label_idxs += obj_seq
                    label_idxs += fake_idxs

                r_text_logits_idxs = list(
                    [idx - (self.l_text_seq_length + self.image_seq_length) for idx in label_idxs])

                loss_r_text = F.cross_entropy(
                    r_text_logits[:, :, r_text_logits_idxs],
                    labels[:, label_idxs],
                    ignore_index=0)
            else:
                loss_r_text = F.cross_entropy(
                    r_text_logits,
                    labels[:, -(self.r_text_seq_length - 1):],
                    ignore_index=0,
                )
            loss_values['r_text_loss'] = loss_r_text.data.detach().float()

            if rt_loss_weight:
                loss += loss_r_text * rt_loss_weight
                loss_weights += rt_loss_weight

            if category_weight > 0:
                category_idxs = list([-(self.r_text_seq_length - 1) + od_seq_len * k + cat_idx
                                      for k in range(self.r_text_seq_length // od_seq_len)])
                rt_category_idxs = list([od_seq_len * k + cat_idx for k in range(self.r_text_seq_length // od_seq_len)])
                loss_category = F.cross_entropy(
                    r_text_logits[:, :, rt_category_idxs],
                    labels[:, category_idxs],
                    ignore_index=0
                )
                loss += loss_category * category_weight
                loss_weights += category_weight

            if loc_loss_weight > 0:
                sum_loc_loss = 0
                right_text_labels_start = labels.size(1) - self.r_text_seq_length
                label_end_idxs = torch.where(labels[0, right_text_labels_start:] == fake_category_token)[0]
                last_object_idx = min(label_end_idxs) - od_seq_len
                if last_object_idx > 0:
                    for object_idx in range(last_object_idx // od_seq_len):
                        for token_idx in range(loc_len):
                            object_start_idx = object_idx * od_seq_len
                            prediction = r_text_logits[:, zero_loc_token:
                                                          zero_loc_token + self.image_tokens_per_dim,
                                         object_start_idx + token_idx]
                            target_loc = labels[
                                             0, right_text_labels_start + r_text_shift + object_start_idx + token_idx] - zero_loc_token
                            loc_loss = self.calc_loc_loss(target_loc, prediction)
                            sum_loc_loss += loc_loss
                    if sum_loc_loss != 0:
                        sum_loc_loss /= loc_len * last_object_idx // od_seq_len
                        loss += sum_loc_loss[0, 0] * loc_loss_weight
                        loss_weights += loc_loss_weight
            if nms_loss_weight > 0 or conf_loss_weight > 0:
                diff_boxes = {}
                prediction = self.make_loc_prediction(r_text_logits[:, :, 1:],
                                                      zero_loc_token,
                                                      categories_num,
                                                      od_seq_len,
                                                      loc_len,
                                                      cat_idx).detach()
                double_idxs = []
                for k in range(prediction.size(-1) // od_seq_len):
                    curr_cat = str(prediction[k * od_seq_len + cat_idx].int().cpu().numpy())
                    curr_box = prediction[k * od_seq_len:k * od_seq_len + loc_len]
                    if curr_cat != str(fake_category_token - zero_loc_token):
                        if curr_cat not in diff_boxes.keys():
                            diff_boxes[curr_cat] = [curr_box]
                        else:
                            unique = True
                            for box in diff_boxes[curr_cat]:
                                iou = self.get_iou(bin_size * box, bin_size * curr_box)
                                if iou > iou_threshold:
                                    double_idxs.append(k * od_seq_len + cat_idx)
                                    unique = False
                                    break
                            if unique:
                                diff_boxes[curr_cat].append(curr_box)
                if len(double_idxs) > 0:
                    nms_tgt = fake_category_token * torch.ones((1, len(double_idxs))).to(self.device)
                    loss_nms = F.cross_entropy(
                        r_text_logits[:, :, double_idxs],
                        nms_tgt.long(),
                        ignore_index=0
                    )
                    loss += loss_nms * nms_loss_weight
                    loss_weights += nms_loss_weight
            if conf_loss_weight > 0 and conf_idx is not None:
                prediction = self.make_loc_prediction(r_text_logits[:, :, 1:],
                                                      zero_loc_token,
                                                      categories_num,
                                                      od_seq_len,
                                                      loc_len,
                                                      cat_idx).detach()
                conf_tokens = []
                conf_idxs = []
                for object_idx in range(prediction.size(-1) // od_seq_len):
                    object_start_idx = object_idx * od_seq_len
                    pred_cat = str(prediction[object_idx * od_seq_len + cat_idx].int().cpu().numpy())
                    tgt_cat = labels[
                        0, right_text_labels_start + r_text_shift + object_start_idx + conf_idx].int().cpu().numpy()
                    if pred_cat != str(fake_category_token - zero_loc_token) and str(tgt_cat) != str(
                            fake_category_token - zero_loc_token):
                        pred_box = prediction[object_idx * od_seq_len:object_idx * od_seq_len + loc_len].to(self.device)
                        tgt_box = labels[0, right_text_labels_start + r_text_shift + object_start_idx:
                                            right_text_labels_start + r_text_shift + object_start_idx + loc_len].float().to(
                            self.device)
                        tgt_box -= zero_loc_token * torch.ones(tgt_box.size()).to(self.device)
                        iou = self.get_iou(bin_size * tgt_box, bin_size * pred_box) * 0.9999
                        iou_bin_num = int(iou * len(iou_bin_tokens))
                        iou_bin_token = iou_bin_tokens[iou_bin_num]
                        conf_tokens.append(iou_bin_token)
                        conf_idxs.append(object_idx * od_seq_len + conf_idx)

                if len(conf_idxs) > 0:
                    conf_tgt = torch.Tensor(conf_tokens).to(self.device).long().unsqueeze(0).detach()
                    loss_conf = F.cross_entropy(
                        r_text_logits[:, :, conf_idxs],
                        conf_tgt,
                        ignore_index=0
                    )
                    loss += loss_conf * conf_loss_weight
                    loss_weights += conf_loss_weight

        loss = loss / loss_weights
        outputs = (loss, loss_values)
        if return_hidden_states:
            outputs += (hidden_states,)
        return outputs

    def make_loc_prediction(self,
                            r_text_logits,
                            zero_loc_token,
                            categories_num,
                            od_seq_len,
                            loc_len,
                            cat_idx):
        pred_len = r_text_logits.size(2) - 1
        prediction = torch.zeros(pred_len)
        for k in range(pred_len // od_seq_len):
            prediction[k * od_seq_len:k * od_seq_len + loc_len] = torch.argmax(
                r_text_logits[:, zero_loc_token:zero_loc_token + self.image_tokens_per_dim,
                k * od_seq_len:k * od_seq_len + loc_len], dim=1)[0]
            prediction[k * od_seq_len + cat_idx] = torch.argmax(
                r_text_logits[:, zero_loc_token + self.image_tokens_per_dim:
                                 zero_loc_token + self.image_tokens_per_dim + categories_num,
                k * od_seq_len + cat_idx], dim=1)[0]+torch.Tensor([self.image_tokens_per_dim]).to(self.device)
        return prediction

    def get_iou(self, a_box, b_box):
        a_x1, a_y1, a_x2, a_y2 = a_box[0], a_box[1], a_box[2], a_box[3]
        b_x1, b_y1, b_x2, b_y2 = b_box[0], b_box[1], b_box[2], b_box[3]

        a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
        b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
        i_x1 = max(a_x1, b_x1)
        i_x2 = min(a_x2, b_x2)
        i_y1 = max(a_y1, b_y1)
        i_y2 = min(a_y2, b_y2)
        intersection = 0
        if i_x2 > i_x1 and i_y2 > i_y1:
            intersection = (i_x2 - i_x1) * (i_y2 - i_y1)

        union = a_area + b_area - intersection
        iou = float(intersection) / float(max(union, 1))

        return iou

    def get_last_bbox_idx(self, labels, fake_category_token, od_seq_len):
        right_text_labels_start = labels.size(1) - self.r_text_seq_length
        label_end_idxs = torch.where(labels[0, right_text_labels_start:] == fake_category_token)[0]
        last_object_idx = min(label_end_idxs) - od_seq_len
        return last_object_idx, right_text_labels_start

    def calc_loc_loss(self, target_loc, prediction):
        prediction -= prediction.min().detach()
        loc_loss_mat = self.loc_loss_matrix[:, target_loc]
        out = torch.mm(prediction, loc_loss_mat.unsqueeze(1))
        return out

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)
