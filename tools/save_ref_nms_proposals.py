import json
import pickle
import argparse
from multiprocessing import Pool

import torch
from tqdm import tqdm
from torchvision.ops import nms
from torch.nn.utils.rnn import pack_padded_sequence

from lib.predictor import AttVanillaPredictorV2
from lib.vanilla_utils import DetEvalLoader
from utils.constants import EVAL_SPLITS_DICT

sub_args = (idx, args.gpu_id, args.tid, refdb_path, split, args.m)
def rank_proposals(position, gpu_id, tid, refdb_path, split, m):
    # Load refdb
    with open(refdb_path) as f:
        refdb = json.load(f)
    dataset_ = refdb['dataset_splitby'].split('_')[0]
    # Load pre-trained model
    device = torch.device('cuda', gpu_id)
    with open('output/{}_{}.json'.format(m, tid), 'r') as f:
        model_info = json.load(f)
    print(model_info)
    predictor = AttVanillaPredictorV2(att_dropout_p=model_info['config']['ATT_DROPOUT_P'],
                                      rank_dropout_p=model_info['config']['RANK_DROPOUT_P'])
    model_path = 'output/{}_{}_b.pth'.format(m, tid)
    #model_path = 'output/att_vanilla_ckpt_0526175202_5.pth'
    print(model_path)
    predictor.load_state_dict(torch.load(model_path))
    predictor.to(device)
    predictor.eval()
    # Rank proposals
    exp_to_proposals = {}
    
    refnmsdet_path, head_feats_dir = 'data/refer/rsvg/refnms_det_instances_rsvg.json', 'data/refer/rsvg/hbb_obb_features_refnms_det'
    det_file_suffix = 'hbb_det_res50_dota_v1_0_RoITransformer.hdf5'
    loader = DetEvalLoader(refdb, split, gpu_id, refnmsdet_path, head_feats_dir, det_file_suffix)
    tqdm_loader = tqdm(loader, desc='scoring {}'.format(split), ascii=True, position=position)
    for exp_id, pos_feat, sent_feat, pos_box, pos_score, pos_ids in tqdm_loader:
        # Compute rank score
        packed_sent_feats = pack_padded_sequence(sent_feat, torch.tensor([sent_feat.size(1)]),
                                                 enforce_sorted=False, batch_first=True)
        with torch.no_grad():
            rank_score, *_ = predictor(pos_feat, packed_sent_feats)  # [1, *]
        pos_feat, sent_feat, pos_box, pos_score, pos_ids, rank_score = pos_feat[0], sent_feat[0], pos_box[0], pos_score[0], pos_ids[0], rank_score[0]
        #print(exp_id)
        #print(pos_feat.shape)
        #print(sent_feat.shape)
        #print(pos_box.shape)
        #print(pos_score.shape)
        #print(pos_ids.shape)
        #print(rank_score.shape)
        # Normalize rank score
        rank_score = torch.sigmoid(rank_score)
        final_score = rank_score * pos_score
        keep = nms(pos_box, final_score, iou_threshold=0.3) #0.3
        kept_box = pos_box[keep]
        kept_score = final_score[keep]
        kept_boxid = pos_ids[keep]
        proposals = []
        for box, score, boxid in zip(kept_box, kept_score, kept_boxid):
            proposals.append({'score': score.item(), 'box': box.tolist(), 'det_box_id': boxid})

        exp_to_proposals[exp_id] = proposals
    return exp_to_proposals


def error_callback(e):
    print('\n\n\n\nERROR in subprocess:', e, '\n\n\n\n')


def main(args):
    dataset_splitby = '{}'.format(args.dataset)
    eval_splits = EVAL_SPLITS_DICT[dataset_splitby]
    refdb_path = 'cache/std_refdb_{}.json'.format(dataset_splitby)
    print('about to rank proposals via multiprocessing, good luck ~')
    results = {}
    with Pool(processes=len(eval_splits)) as pool:
        for idx, split in enumerate(eval_splits):
            sub_args = (idx, args.gpu_id, args.tid, refdb_path, split, args.m)
            results[split] = pool.apply_async(rank_proposals, sub_args, error_callback=error_callback)
        pool.close()
        pool.join()
    proposal_dict = {}
    for split in eval_splits:
        assert results[split].successful()
        print('subprocess for {} split succeeded, fetching results...'.format(split))
        proposal_dict[split] = results[split].get()
    save_path = 'cache/proposals_{}_{}_{}.pkl'.format(args.m, args.dataset, args.tid)
    print('saving proposals to {}...'.format(save_path))
    with open(save_path, 'wb') as f:
        pickle.dump(proposal_dict, f)
    print('all done ~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--dataset', default='rsvg')
    parser.add_argument('--tid', type=str, required=True)
    parser.add_argument('--m', type=str, default='att_vanilla')
    main(parser.parse_args())
