import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import copy
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
from einops import repeat
import itertools, math
import pdb 


''' video_tensor = vid_feats -> tensor
    video_length = video_length -> tensor
    video_target is a list of all the videos in a batch, size(target)=batch_size
    it contains 'boxes', and 'image ids' 
    gt_timestamps = gt timestamps for each event in a video
    cap_tensor = caption tensor of all events with time token together
    cap_length = capion length mask
'''
class Vid2Seq(nn.Module):
    def __init__(self, cfg, tokenizer, vct_encoder, T5_enc_dec, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.vct_context = vct_encoder
        self.T5_enc_dec = T5_enc_dec
        self.device = device
    
    def forward(self,inp, eval_mode=False):
        encoder_outputs = self.vct_context(inp, eval_mode=eval_mode) #BxNxD
        out, loss = self.T5_enc_dec(inp, encoder_outputs, eval_mode=eval_mode)
        return out, loss
    
    def forward_gen(self, inp, eval_mode=False):
        encoder_outputs = self.vct_context(inp, eval_mode=eval_mode) #BxNxD
        outputs = self.T5_enc_dec.forward_gen(inp, encoder_outputs, self.device)
        return outputs



class T5_encoder_decoder(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, inp, vid_encoder_out, eval_mode):
        caption_tensor = inp['cap_tensor'] #BxC
        caption_length = inp['cap_length'] # bool 

        vid_encoder_out = (vid_encoder_out, (), ())  #vid_encoder_out -> (BxNxD, (), ())
        caption_tensor[caption_tensor == self.tokenizer.pad_token_id] = -100
        output = self.model(encoder_outputs=vid_encoder_out, labels=caption_tensor)
        loss = output.loss
        return output, loss

    def forward_gen(self, inp, encoder_out, device):
        B,N,D = encoder_out.size()
        encoder_input_ids = torch.LongTensor([[self.tokenizer.pad_token_id]]).to(device)
        # decoder_input_ids=decoder_input_ids.to(device)
        # Todo: Hacky find a better way # fake forward pass through the encoder to extract dummy encoder_output object
        dummy_enc_out = self.model.encoder(input_ids=encoder_input_ids)
        outputs = []
        captions = []
        for vid_idx in range(B):    
            dummy_enc_out['last_hidden_state'] = encoder_out[vid_idx].unsqueeze(0)
            out = self.model.generate(input_ids=encoder_input_ids,
                    encoder_outputs=dummy_enc_out,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    num_return_sequences=1,
                )
            outputs.append(out[0])
            # out_cap = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # captions.append(out_cap)
        # outputs = torch.cat(outputs, dim=0)
        return outputs
        

class VideoContextualizer(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.dim = 768
        self.dim_ff = 2048
        num_layers = 3
        dropout = 0.15
        activation= "relu"
        self.tokenizer = tokenizer
        self.pos_embed = nn.Embedding(100, self.dim)
        self.project_vid = nn.Sequential(nn.Linear(512, self.dim),
                                             nn.ReLU(),
                                             nn.Linear(self.dim, self.dim))
        nhead = 8

        # Video context Transformer encoder
        vct_encoder_layer = TransformerEncoderLayer(self.dim, nhead, self.dim_ff, dropout, activation)
        vct_encoder_norm = nn.LayerNorm(self.dim)
        self.vct_encoder = TransformerEncoder(vct_encoder_layer, num_layers, vct_encoder_norm)

    def forward(self, inp, eval_mode):  
        vid_feats = inp['video_tensor'] #BxNx512
        B,N,D = vid_feats.size()
        vid_embs = self.project_vid(vid_feats)
        len_vid = N
        #position embeddings
        positions = vid_feats.new(N).long() 
        positions = torch.arange(N, out=positions) 
        pos_embs = repeat(self.pos_embed(positions), 'n d -> b n d', b = B) # Bx5xD 
        vct_enc_inp_emb = vid_embs+pos_embs
        #attention mask
        alen = vid_feats.new(len_vid).long() 
        alen = torch.ones(len_vid, out=alen) 
        vid_atten_mask = alen[None,:].repeat(B,1)

        #feed the processed inputs to the vct_encoder
        vid_obj_out = self.vct_encoder(vct_enc_inp_emb, src_key_padding_mask=vid_atten_mask)
        return vid_obj_out
    



class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)        

    def forward(self, query, key=None, value=None, mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = query.size()
        if key is None:
            klen = qlen 
        else:
            klen = key.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))                                          # (bs, n_heads, qlen, dim_per_head)
        if key is None:
            k = shape(self.k_lin(query))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(query))                                      # (bs, n_heads, qlen, dim_per_head)
        else:
            k = key
            v = value
            k = shape(self.k_lin(k))                                          # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))                                          # (bs, n_heads, qlen, dim_per_head)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)
        
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)
        
        attn_scores=weights.sum(dim=1)/n_heads
        attn_scores_max=torch.argmax(attn_scores, dim=-1)
        return self.out_lin(context), attn_scores_max
    


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_key_padding_mask: Optional[Tensor] = None):
        src2 = self.norm1(src)
        src2, src_attn_out = self.self_attn(query=src2, mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        src = src + self.dropout2(src2)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    return m



# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""

#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt
#         if opt.enable_contrastive and vars(opt).get('eval_enable_grounding', False):
#             from pdvc.matcher import HungarianMatcher
#             self.grounding_matcher = HungarianMatcher(cost_class=opt.eval_set_cost_class,
#                             cost_bbox=0.0,
#                             cost_giou=0.0,
#                             cost_alpha = opt.eval_grounding_cost_alpha,
#                             cost_gamma = opt.eval_grounding_cost_gamma,
#                             cost_cl= opt.eval_set_cost_cl,
#                             )

#     @torch.no_grad()
#     def forward_grounding(self, outputs, target_sizes, targets):
#         if not self.opt.enable_contrastive:
#             return None, None

#         for target in targets:
#             target['boxes'] = target['boxes'] * 0
#             target['labels'] = target['labels'] * 0

#         all_boxes = box_ops.box_cl_to_xy(outputs['pred_boxes'])
#         all_boxes[all_boxes < 0] = 0
#         all_boxes[all_boxes > 1] = 1
#         scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
#         all_boxes = all_boxes * scale_fct[:, None, :]
#         all_boxes = all_boxes.cpu().numpy().tolist()

#         all_logits = outputs['pred_logits'].sigmoid().cpu().numpy().tolist()
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
#         last_indices,_,C = self.grounding_matcher(outputs_without_aux, targets, return_C=True)

#         def get_results(indices, C):
#             results = []
#             for i, (event_ind, cap_ind) in enumerate(indices):
#                 N_cap = len(targets[i]['boxes'])
#                 boxes = []
#                 confs = []
#                 cl_scores = []
#                 cap_ind = cap_ind.numpy().tolist()
#                 for j in range(N_cap):
#                     if self.opt.eval_enable_maximum_matching_for_grounding:
#                         event_j = C[i][:, j].argmin()
#                     else:
#                         if j not in cap_ind:
#                             # print(C[0].shape, len(C), j)
#                             event_j = C[i][:, j].argmin()
#                         else:
#                             match_id = cap_ind.index(j)
#                             event_j = event_ind[match_id]
#                     boxes.append(all_boxes[i][event_j])
#                     confs.append(all_logits[i][event_j][0])
#                     cl_scores.append(C[i][event_j, j].item())
#                 results.append({'boxes': boxes, 'confs': confs, 'cl_scores': cl_scores})
#             return results

#         last_results = get_results(last_indices, C)
#         cl_scores = outputs['cl_match_mats']
#         sizes = [len(v["boxes"]) for v in targets]
#         if cl_scores.shape[1] > sum(sizes):
#             bs, num_queries, _ = outputs['pred_boxes'].shape
#             bg_cl_score = cl_scores[:, -1:].reshape(bs, num_queries, 1)
#             cl_scores = cl_scores[:, :-1].reshape(bs, num_queries, -1)
#             cl_scores = [torch.cat((c[i], bg_cl_score[i]), dim=1) for i, c in enumerate(cl_scores.split(sizes, dim=-1))]
#         return last_results, cl_scores

#     @torch.no_grad()
#     def forward(self, outputs, target_sizes, loader, model=None, tokenizer=None):
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
#         """
#         out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
#         N, N_q, N_class = out_logits.shape
#         assert len(out_logits) == len(target_sizes)

#         prob = out_logits.sigmoid()
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
#         scores = topk_values
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         labels = topk_indexes % out_logits.shape[2]
#         boxes = box_ops.box_cl_to_xy(out_bbox)
#         raw_boxes = copy.deepcopy(boxes)
#         boxes[boxes < 0] = 0
#         boxes[boxes > 1] = 1
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

#         scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
#         boxes = boxes * scale_fct[:, None, :]
#         seq = outputs['seq']  # [batch_size, num_queries, max_Cap_len=30]
#         cap_prob = outputs['caption_probs']['cap_prob_eval']  # [batch_size, num_queries]
#         eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)
#         bs, num_queries = boxes.shape[:2]

#         if seq is None and 'gpt2_cap' in outputs['caption_probs']:
#             caps = outputs['caption_probs']['gpt2_cap']
#             cap_idx = 0
#             caps_new = []
#             for batch, b in enumerate(topk_boxes):
#                 caps_b = []
#                 for q_id, idx in enumerate(b):
#                     caps_b.append(caps[cap_idx])
#                     cap_idx += 1
#                 caps_new.append(caps_b)
#             caps = caps_new
#             mask = outputs['caption_probs']['gen_mask']
#             cap_prob = outputs['caption_probs']['cap_prob_eval']
#             cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
#             caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
#         else:
#             if len(seq):
#                 mask = (seq > 0).float()
#                 cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
#                 seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
#                 caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
#                 caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
#                 cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
#             else:
#                 bs, num_queries = boxes.shape[:2]
#                 cap_scores = [[-1e5] * num_queries] * bs
#                 caps = [[''] * num_queries] * bs

#         if self.opt.enable_contrastive and self.opt.eval_enable_matching_score:
#             event_embed = outputs['event_embed']
#             cap_list = list(chain(*caps))
#             text_encoder_inputs = tokenizer(cap_list, return_tensors='pt', padding=True)

#             text_encoder_inputs = {key: _.to(self.opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in
#                                   text_encoder_inputs.items()}

#             input_cap_num = [len(_) for _ in caps]
#             memory = outputs.get('memory', [None] * len(input_cap_num))
#             text_embed, word_embed, _, _ = model.text_encoding(text_encoder_inputs, input_cap_num, memory=memory)

#             text_embed = torch.cat(text_embed[-1], dim=0) # feature of last decoder layer
#             event_embed = event_embed.reshape(-1, event_embed.shape[-1])

#             normalized_text_emb = F.normalize(text_embed, p=2, dim=1)
#             normalized_event_emb = F.normalize(event_embed, p=2, dim=1)
#             cl_logits = torch.mm(normalized_text_emb, normalized_event_emb.t())

#             sizes = [num_queries] * bs
#             cl_pre_logit = [torch.eq(m.split(sizes, 0)[i].argmax(dim=1), topk_indexes[i]).sum() for i, m in enumerate(cl_logits.split(sizes, 1))]
#             cl_scores = [torch.gather(m.split(sizes, 0)[i], 1, topk_indexes[i].unsqueeze(1)).squeeze(1) for i, m in enumerate(cl_logits.split(sizes, 1))]
#             cl_scores = [cl_score.cpu().numpy().astype('float') for cl_score in cl_scores]
#         else:
#             cl_scores = [[0.0] * num_queries] * bs

#         results = [
#             {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'cl_scores': cls,'query_id': qid,
#              'vid_duration': ts, 'pred_seq_len': sl, 'raw_idx': idx} for s, l, b, rb, c, cs, cls, qid, ts, sl, idx in
#             zip(scores, labels, boxes, raw_boxes, caps, cap_scores, cl_scores, topk_boxes, target_sizes, eseq_lens, topk_indexes)]
#         return results
    

def build(args, tokenizer, device):
    # device = torch.device(args.device)
    vct_encoder = VideoContextualizer(tokenizer)
    T5_enc_dec = T5_encoder_decoder(tokenizer)

    model = Vid2Seq(
        args,
        tokenizer,
        vct_encoder,
        T5_enc_dec,
        device
    )

    # postprocessors = {'bbox': PostProcess(args)}

    return model
