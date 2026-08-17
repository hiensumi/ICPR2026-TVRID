# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        pids = inputs['targets']
        if isinstance(pids, torch.Tensor):
            pids = pids.to(self._cpu_device)
        
        camids = inputs['camids']
        if isinstance(camids, torch.Tensor):
            camids = camids.to(self._cpu_device)

        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': pids,
            'camids': camids
        }
        if 'passage_ids' in inputs:
            prediction['passage_ids'] = inputs['passage_ids']
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        passage_ids = []
        has_passage_ids = all('passage_ids' in prediction for prediction in predictions)
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            if has_passage_ids:
                passage_ids.append(prediction['passage_ids'])

        features = torch.cat(features, dim=0)
        
        # Handle pids concatenation
        if len(pids) > 0 and isinstance(pids[0], torch.Tensor):
            pids = torch.cat(pids, dim=0).numpy()
        else:
            # Flatten list of lists/tensors
            pids = list(itertools.chain(*pids))
            pids = np.array(pids)

        # Handle camids concatenation
        if len(camids) > 0 and isinstance(camids[0], torch.Tensor):
            camids = torch.cat(camids, dim=0).numpy()
        else:
            camids = list(itertools.chain(*camids))
            camids = np.array(camids)

        # Label Encode PIDs and CamIDs to integers if they are strings (object type)
        if pids.dtype == object or pids.dtype.type is np.str_ or pids.dtype.type is np.bytes_:
            from sklearn.preprocessing import LabelEncoder
            pids = LabelEncoder().fit_transform(pids)

        if camids.dtype == object or camids.dtype.type is np.str_ or camids.dtype.type is np.bytes_:
            from sklearn.preprocessing import LabelEncoder
            camids = LabelEncoder().fit_transform(camids)

        num_query = self._num_query
        if has_passage_ids:
            passage_ids = np.array(list(itertools.chain(*passage_ids)))

            def _aggregate_by_passage(feats, pid_arr, cam_arr, passage_arr):
                order = OrderedDict()
                for idx, passage_id in enumerate(passage_arr):
                    order.setdefault(passage_id, []).append(idx)

                agg_feats = []
                agg_pids = []
                agg_camids = []
                for idxs in order.values():
                    idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=feats.device)
                    feat = feats.index_select(0, idx_tensor).mean(dim=0)
                    agg_feats.append(feat)
                    agg_pids.append(pid_arr[idxs[0]])
                    agg_camids.append(cam_arr[idxs[0]])

                agg_feats = F.normalize(torch.stack(agg_feats, dim=0), dim=1)
                return agg_feats, np.asarray(agg_pids), np.asarray(agg_camids)

            q_feats, q_pids, q_camids = _aggregate_by_passage(
                features[:self._num_query],
                pids[:self._num_query],
                camids[:self._num_query],
                passage_ids[:self._num_query],
            )
            g_feats, g_pids, g_camids = _aggregate_by_passage(
                features[self._num_query:],
                pids[self._num_query:],
                camids[self._num_query:],
                passage_ids[self._num_query:],
            )
            features = torch.cat([q_feats, g_feats], dim=0)
            pids = np.concatenate([q_pids, g_pids], axis=0)
            camids = np.concatenate([q_camids, g_camids], axis=0)
            num_query = len(q_pids)

        # query feature, person ids and camera ids
        query_features = features[:num_query]
        query_pids = pids[:num_query]
        query_camids = camids[:num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[num_query:]
        gallery_pids = pids[num_query:]
        gallery_camids = camids[num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            # logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        t0 = time.perf_counter()
        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)
        logger.info("dist ({}) time: {:.1f}s".format(self.cfg.TEST.METRIC, time.perf_counter() - t0))

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            t0 = time.perf_counter()
            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            logger.info("rerank (jaccard) time: {:.1f}s".format(time.perf_counter() - t0))
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        # Filter queries that do not have any match in the gallery
        # This prevents "AssertionError: Error: all query identities do not appear in gallery"
        unique_gallery_pids = set(gallery_pids)
        valid_query_mask = np.array([qp in unique_gallery_pids for qp in query_pids])
        
        if not np.all(valid_query_mask):
            logger.warning(f"Filtering {np.sum(~valid_query_mask)} queries that do not appear in gallery.")
            query_features = query_features[valid_query_mask]
            query_pids = query_pids[valid_query_mask]
            query_camids = query_camids[valid_query_mask]
            dist = dist[valid_query_mask] # CRITICAL: Also filter distance matrix rows!
            
            # Verify we still have queries
            if len(query_pids) == 0:
                logger.error("No valid queries remaining after filtering! Returning empty metrics.")
                return {}

        # Ensure types are correct for Cython
        query_pids = np.asarray(query_pids).astype(np.int64)
        gallery_pids = np.asarray(gallery_pids).astype(np.int64)
        query_camids = np.asarray(query_camids).astype(np.int64)
        gallery_camids = np.asarray(gallery_camids).astype(np.int64)
        dist = np.asarray(dist).astype(np.float32)

        from .rank import evaluate_rank
        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                try:
                    compile_helper()
                except Exception as e:
                    logger.warning(f"Failed to compile Cython module: {e}. Falling back to Python implementation.")
                    return

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()
