from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import sys
import collections
import torch
import numpy as np
import json
from collections import OrderedDict
from tqdm import tqdm
from os.path import dirname, abspath
from transformers import AutoTokenizer
import math

pdvc_root_dir = dirname(abspath(__file__))

for pdvc_dir in [pdvc_root_dir]:
    sys.path.insert(0, pdvc_dir)
    sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
    sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))


from densevid_eval3.eval_soda import eval_soda
from densevid_eval3.eval_para import eval_para
from densevid_eval3.eval_dvc import eval_dvc
from densevid_eval3.eval_tal import eval_tal

# from misc.plot_proposal_distribution import main as plot_proposal_distribution
# from densevid_eval3.eval_grounding import eval_result as eval_grounding

def calculate_avg_proposal_num(json_path):
    data = json.load(open(json_path))
    return np.array([len(v) for v in data['results'].values()]).mean()

def convert_tapjson_to_dvcjson(tap_json, dvc_json):
    data = json.load(open(tap_json, 'r'))
    data['version'] = "VERSION 1.0"
    data['external_data'] = {'used:': True, 'details': "C3D pretrained on Sports-1M"}

    all_names = list(data['results'].keys())
    for video_name in all_names:
        for p_info in data['results'][video_name]:
            p_info['timestamp'] = p_info.pop('segment')
            p_info['proposal_score'] = p_info.pop('score')
            p_info['sentence_score'] = p_info.pop('sentence_score', 0)
        data['results']["v_" + video_name] = data['results'].pop(video_name)
    json.dump(data, open(dvc_json, 'w'))


def convert_dvcjson_to_tapjson(dvc_json, tap_json):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        event_num = len(data[video_name])
        timestamps = [data[video_name][i]['timestamp'] for i in range(event_num)]
        sentences = [data[video_name][i]['sentence'] for i in range(event_num)]
        for i, timestamp in enumerate(timestamps):
            score = data[video_name][i].get('proposal_score', 1.0)
            video_info.append({'segment': timestamp, 'score': score, 'sentence': sentences[i], 'sentence_score': data[video_name][i].get('sentence_score', 0)})
        out['results'][video_name[2:]] = video_info
    json.dump(out, open(tap_json, 'w'))


def convert_gtjson_to_tapjson(gt_json, tap_json):
    data = json.load(open(gt_json, 'r'))
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}

    all_names = list(data.keys())
    for video_name in all_names:
        video_info = []
        timestamps = data[video_name]['timestamps']
        sentences = data[video_name]['sentences']
        for i, timestamp in enumerate(timestamps):
            video_info.append({'segment': timestamp, 'score': 1., 'sentence': sentences[i]})
        out['results'][video_name[2:]] = video_info
    with open(tap_json, 'w') as f:
        json.dump(out, f)


def get_topn_from_dvcjson(dvc_json, out_json, top_n=3, ranking_key='proposal_score', score_thres=-1e8):
    data = json.load(open(dvc_json, 'r'))['results']
    out = {}
    out['version'] = "VERSION 1.0"
    out['external_data'] = {'used:': True, 'details': "GT proposals"}
    out['results'] = {}
    all_names = list(data.keys())
    num = 0
    bad_vid = 0
    for video_name in all_names:
        info = data[video_name]
        new_info = sorted(info, key=lambda x: x[ranking_key], reverse=True)
        new_info = [p for p in new_info if p[ranking_key] > score_thres]
        new_info = new_info[:top_n]
        out['results'][video_name] = new_info
        num += len(new_info)
        if len(new_info) == 0:
            bad_vid += 1
            out['results'].pop(video_name)
    print('average proosal number: {}'.format(num / len(all_names)))
    print('bad videos number: {}'.format(bad_vid))
    print('good videos number: {}'.format(len(out['results'])))
    with open(out_json, 'w') as f:
        json.dump(out, f)

# def eval_metrics_grounding(g_filename, gt_filename):
#     score = collections.defaultdict(lambda: -1)
#     grounding_scores = eval_grounding(g_filename, gt_filename)
#     for key in grounding_scores.keys():
#         score['grounding_'+key] = grounding_scores[key]
#     return score

def eval_metrics(dvc_filename, gt_filenames, para_gt_filenames, alpha=0.3, temperature=2.0, cl_score_weight=0., ranking_key='proposal_score', rerank=False, dvc_eval_version='2018', verbose=False):
    score = collections.defaultdict(lambda: -1)

    dvc_score = eval_dvc(json_path=dvc_filename, reference=gt_filenames, version=dvc_eval_version, verbose=verbose)
    dvc_score = {k: sum(v) / len(v) for k, v in dvc_score.items()}
    dvc_score.update(eval_soda(dvc_filename, ref_list=gt_filenames))
    dvc_score.update(eval_para(dvc_filename, referneces=para_gt_filenames))
    dvc_score.update({'MetaScore': dvc_score['METEOR'] + dvc_score['soda_c']})
    score.update(dvc_score)
    return score


def save_dvc_json(out_json, path, verbose=False):
    with open(path, 'w') as f:
        if verbose:
            out_json['valid_video_num'] = len(out_json['results'])
            out_json['avg_proposal_num'] = np.array([len(v) for v in out_json['results'].values()]).mean().item()
        json.dump(out_json, f)


def extract_all_caps(output, tokenizer):
    event_tokens_word = ["<{}>".format(i) for i in range (100)]
    event_tokens_idx = [tokenizer.encode(event_tokens_word[i])[0] for i in range(100)]

    cap_event_toks = []
    cap_event_toks_pos = []
    for i,token in enumerate(output):
        if token in event_tokens_idx:
            if i+1<len(output)-5:
                token_next = output[i+1]
            if token_next in event_tokens_idx:
                cap_event_toks.append([tokenizer.decode(token),tokenizer.decode(token_next)])
                cap_event_toks_pos.append([i,i+1])
    
    sentences = []
    for j in range(len(cap_event_toks_pos)):
        pos_st = cap_event_toks_pos[j][1]+1
        if j+1 < len(cap_event_toks_pos):
            pos_end = cap_event_toks_pos[j+1][0]
        else:
            pos_end = len(output)-1

        sent_toks = output[pos_st:pos_end]
        sentence = tokenizer.decode(sent_toks, skip_special_tokens=True)
        sentences.append(sentence)
    
    assert len(cap_event_toks) == len(sentences)
    
    return cap_event_toks, sentences
        
        
def extract_timestamps_from_event_tokens(event_toks, vid_duration):
    event_segments = []
    for i, ev_toks in enumerate(event_toks):
            ev_st_tok = ev_toks[0]
            ev_st = int(ev_st_tok[1:-1])
            ev_end_tok = ev_toks[1]
            ev_end = int(ev_end_tok[1:-1])
            event_seg = [ev_st*vid_duration/100., ev_end*vid_duration/100.]
            event_segments.append(event_seg)
    return event_segments



def postprocess(outputs,dt, tokenizer):
    batch_json = {}
    for idx, video_name in enumerate(dt['video_key']):
        all_event_toks, all_caps = extract_all_caps(outputs[idx].cpu().numpy(), tokenizer)
        all_segments = extract_timestamps_from_event_tokens(all_event_toks, dt['video_length'][idx][1].cpu().item())
        assert len(all_segments) == len(all_caps)
    
        batch_json[video_name] = [{"timestamp": all_segments[pid], "sentence": all_caps[pid]} for pid in range(len(all_segments))]
    return batch_json


def evaluate(model, loader, dvc_json_path, logger=None, score_threshold=0,
             alpha=0.3, dvc_eval_version='2018', device='cuda', debug=False, skip_lang_eval=False, verbose=False, tokenizer=None):
    
    out_json = {'results': {},
                'version': "VERSION 1.0",
                'external_data': {'used:': True, 'details': None}}

    out_json_g = {'results': {}}
    aux_out_json_g = {'results': {}}

    opt = loader.dataset.opt
    # Load tokenizer for text encoder
    # if tokenizer is None:
    #     tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_language_model, cache_dir=opt.huggingface_cache_dir)

    loss_sum = OrderedDict()
    with torch.set_grad_enabled(False):
        for dt in tqdm(loader, disable=opt.disable_tqdm):
            dt = {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            # dt = collections.defaultdict(lambda: None, dt)

            dt['video_target'] = [
                    {key: _.to(device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                    dt['video_target']]

            # output_, loss = model(dt, eval_mode=True)
            outputs = model.forward_gen(dt, eval_mode=True)
            batch_json = postprocess(outputs, dt, tokenizer)
            
            # final_loss = 2.3
            loss_sum['total_loss'] = 2.3
                
            out_json['results'].update(batch_json)
    pdb.set_trace()
    save_dvc_json(out_json, dvc_json_path, verbose=True)

    # try:
    #     plot_proposal_distribution(dvc_json_path)
    # except:
    #     pass
    
    for k in loss_sum.keys():
        loss_sum[k] = np.round(loss_sum[k] / (len(loader) + 1e-5), 3).item()
    if logger is not None:
        logger.info('loss: {}'.format(loss_sum))

    # if opt.count_loss_coef > 0:
    #     dvc_json_path = reranking(dvc_json_path, alpha=alpha, cl_score_weight=opt.eval_matching_score_weight, temperature=2.0)
    # save_dvc_json(out_json_g, dvc_json_path + '.grounding.json')
    # save_dvc_json(aux_out_json_g, dvc_json_path + '_aux.grounding.json')

    scores = eval_metrics(dvc_json_path,
                            gt_filenames=opt.gt_file_for_eval,
                            para_gt_filenames=opt.gt_file_for_para_eval,
                            alpha=alpha,
                            cl_score_weight=opt.eval_matching_score_weight,
                            rerank=(opt.count_loss_coef > 0),
                            dvc_eval_version=dvc_eval_version,
                            verbose=verbose
                            )
    
    out_json.update(scores)
    # scores_g = eval_metrics_grounding(dvc_json_path + '.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)
    # aux_scores_g = eval_metrics_grounding(dvc_json_path + '_aux.grounding.json', gt_filename=opt.eval_gt_file_for_grounding)
    # rename_aux_scores_g = {'aux_' + key: value for key, value in aux_scores_g.items()}
    # out_json_g.update(scores_g)
    # aux_out_json_g.update(aux_scores_g)
    # scores.update(scores_g)
    # scores.update(rename_aux_scores_g)
    # if opt.only_ft_class_head:
    #     score_tal = eval_tal(ground_truth_filename=opt.tal_gt_file, prediction_filename=tal_result_json_path)
    #     out_json_tal.update(score_tal)
    #     save_dvc_json(out_json_tal, tal_result_json_path)
    #     scores.update(score_tal) 
    # save_dvc_json(out_json, dvc_json_path, verbose=True)
    # save_dvc_json(out_json_g, dvc_json_path + '.grounding.json')
    # save_dvc_json(aux_out_json_g, dvc_json_path + '_aux.grounding.json')
    return scores, loss_sum


