from __future__ import print_function, absolute_import
import sys
import time
import torch
from collections import OrderedDict
from .evaluation_metrics import cmc, mean_ap

def extract_cnn_feature(model, inputs):
    model = model.eval()
    with torch.inference_mode():
        outputs = model(inputs.cuda())
    return outputs

def extract_features(model, data_loader, verbose=False):
    fea_time = 0
    data_time = 0
    features = OrderedDict()
    end = time.time()

    if verbose:
        print('Extract Features...')

    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time += time.time() - end
        end = time.time()
        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
        fea_time += time.time() - end
        end = time.time()
    if verbose:
        print('Feature time: {:.3f} seconds. Data time: {:.3f} seconds.'.format(fea_time, data_time))

    return features


def pairwise_distance(matcher, prob_fea, gal_fea, gal_batch_size=128, prob_batch_size=128):
    with torch.inference_mode():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score = torch.zeros(num_probs, num_gals, device=prob_fea.device)
        matcher.eval()
        for i in range(0, num_probs, prob_batch_size):
            j = min(i + prob_batch_size, num_probs)
            matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                score[i: j, k: k2] = matcher(gal_fea[k: k2, :, :, :].cuda())

        dist = -score
        dist = dist.cpu().float()
    return dist  # [p, g]


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    # print('CMC Scores')
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}'
    #           .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model, matcher):
        super(Evaluator, self).__init__()
        self.model = model
        self.matcher = matcher

    def evaluate(self, testset, query_loader, gallery_loader):
        query = testset.query
        gallery = testset.gallery

        print('Compute similarity ...')
        start = time.time()

        prob_fea = extract_features(self.model, query_loader)
        prob_fea = torch.cat([prob_fea[f].unsqueeze(0) for f, _, _, _ in query], 0)
        num_prob = len(query)
        num_gal = len(gallery)
        batch_size = gallery_loader.batch_size
        dist = torch.zeros(num_prob, num_gal)

        for i, (imgs, fnames, pids, _) in enumerate(gallery_loader):
            print('Extract features... %d / %d. \t' % (i + 1, len(gallery_loader)), end='\r', file=sys.stdout.console)
            gal_fea = extract_cnn_feature(self.model, imgs)
            g0 = i * batch_size
            g1 = min(num_gal, (i + 1) * batch_size)
            dist[:, g0:g1] = pairwise_distance(self.matcher, prob_fea, gal_fea)  # [p, g]

        print('Time: %.3f seconds.' % (time.time() - start))
        rank1, mAP = evaluate_all(dist, query=query, gallery=gallery)

        return rank1, mAP