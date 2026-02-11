from __future__ import absolute_import, division, print_function

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from PIL import Image

from datasets import AntiUAV410
from utils.metrics import rect_iou, center_error
from utils.viz import show_frame


class ExperimentAntiUAV410(object):
    r"""Experiment pipeline and evaluation toolkit for AntiUAV410 dataset."""

    def __init__(self, root_dir, subset,
                 result_dir='results', report_dir='reports', start_idx=0, end_idx=None):
        super(ExperimentAntiUAV410, self).__init__()
        self.subset = subset
        self.dataset = AntiUAV410(os.path.join(root_dir, subset))
        self.result_dir = os.path.join(result_dir, 'AntiUAV410', subset)
        self.report_dir = os.path.join(report_dir, 'AntiUAV410', subset)
        # as nbins_iou increases, the success score converges to AO
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.use_confs = True
        self.dump_as_csv = False
        self.att_name = ['Thermal Crossover', 'Out-of-View', 'Scale Variation',
                         'Fast Motion', 'Occlusion', 'Dynamic Background Clutter',
                         'Tiny Size', 'Small Size', 'Medium Size', 'Normal Size']
        self.att_fig_name = ['TC', 'OV', 'SV', 'FM', 'OC', 'DBC',
                             'TS', 'SS', 'MS', 'NS']

    # ---------- helpers ----------
    def _valid_mask(self, anno_xywh: np.ndarray, exist_list) -> np.ndarray:
        """exist==1 & (w>0,h>0) 만 유효 프레임으로 사용"""
        anno = np.asarray(anno_xywh, dtype=float)
        exist = np.asarray(exist_list, dtype=bool)[:len(anno)]
        if anno.ndim != 2 or anno.shape[1] != 4:
            return np.zeros((len(anno),), dtype=bool)
        return exist & (anno[:, 2] > 0) & (anno[:, 3] > 0)

    def _pad_to_len(self, arr, T, fill=0.0) -> np.ndarray:
        """Nx4 float 배열을 길이 T로 패딩/절단. 4가 아닌 행은 더미로 대체."""
        arr = list(arr)
        fixed = []
        for r in arr:
            if isinstance(r, (list, tuple, np.ndarray)) and len(r) == 4:
                fixed.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
            else:
                fixed.append([fill, fill, fill, fill])
        A = np.asarray(fixed, dtype=float)
        n = A.shape[0]
        if n < T:
            pad = np.full((T - n, 4), fill, dtype=float)
            return np.vstack([A, pad])
        return A[:T]

    # ---------- run & SA ----------
    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (tracker.name, type(self.dataset).__name__))

        end_idx = self.end_idx if self.end_idx is not None else len(self.dataset)
        overall_performance = []

        for s in range(self.start_idx, end_idx):
            img_files, label_res = self.dataset[s]
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # tracking loop
            bboxes, times = tracker.forward_test(img_files, label_res['gt_rect'][0], visualize=visualize)

            # record results
            self._record(record_file, bboxes, times)
            SA_Score = self.eval(bboxes, label_res)
            overall_performance.append(SA_Score)
            print('%20s Fixed Measure: %.03f' % (seq_name, SA_Score))

        print('[Overall] Mixed Measure: %.03f\n' % (np.mean(overall_performance)))

    def iou(self, bbox1, bbox2):
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        (x0_1, y0_1, w1_1, h1_1) = bbox1
        (x0_2, y0_2, w1_2, h1_2) = bbox2
        x1_1 = x0_1 + w1_1; y1_1 = y0_1 + h1_1
        x1_2 = x0_2 + w1_2; y1_2 = y0_2 + h1_2

        ox0 = max(x0_1, x0_2); oy0 = max(y0_1, y0_2)
        ox1 = min(x1_1, x1_2); oy1 = min(y1_1, y1_2)
        if ox1 - ox0 <= 0 or oy1 - oy0 <= 0:
            return 0.0
        s1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        s2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        inter = (ox1 - ox0) * (oy1 - oy0)
        union = s1 + s2 - inter
        return inter / union if union > 0 else 0.0

    def not_exist(self, pred):
        return 1.0 if (len(pred) in (0, 1)) else 0.0

    def eval(self, out_res, label_res):
        measure_per_frame = []
        for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
            if not _exist:
                measure_per_frame.append(self.not_exist(_pred))
            else:
                if len(_gt) < 4 or sum(_gt) == 0:
                    continue
                if len(_pred) == 4:
                    measure_per_frame.append(self.iou(_pred, _gt))
                else:
                    measure_per_frame.append(0.0)
        return np.mean(measure_per_frame)

    # ---------- report (OPE) ----------
    def report(self, trackers, plot_curves=True, plot_attcurves=True):
        assert isinstance(trackers, (list, tuple))
        if not isinstance(trackers[0], dict):
            trackers = [{'name': trackers[0],
                         'path': os.path.join(self.result_dir, trackers[0]),
                         'mode': 1}]

        report_dir = os.path.join(self.report_dir)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        tracker_names = []
        overall_SAs = []

        for trackerid, tracker in enumerate(trackers):
            name = tracker['name']; mode = tracker['mode']
            tracker_names.append(name)
            print('Evaluating', name)

            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou), dtype=float)
            prec_curve = np.zeros((seq_num, self.nbins_ce), dtype=float)
            speeds = np.zeros(seq_num, dtype=float)

            performance.update({name: {'overall': {},
                                       'TC': {}, 'OV': {}, 'SV': {}, 'FM': {}, 'OC': {}, 'DBC': {},
                                       'TS': {}, 'SS': {}, 'MS': {}, 'NS': {},
                                       'seq_wise': {}}})

            overall_SA = []
            att_list = []

            for s, (_, label_res) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(tracker['path'], '%s.txt' % seq_name)

                if name in ('SwinTrack-Tiny', 'SwinTrack-Base'):
                    record_file = os.path.join(tracker['path'],
                                               'test_metrics/anti-uav410-%s/%s/bounding_box.txt' % (self.subset, seq_name))

                # attributes
                att_file = os.path.join('annos', self.subset, 'att', '%s.txt' % seq_name)
                with open(att_file, 'r') as f:
                    att_temp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
                att_list.append(att_temp)

                # load preds
                try:
                    with open(record_file, 'r') as f:
                        boxestemp = json.load(f)['res']
                except Exception:
                    with open(record_file, 'r') as f:
                        boxestemp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

                if mode == 2:
                    boxestemp[:, 2:] = boxestemp[:, 2:] - boxestemp[:, :2] + 1  # XYXY->XYWH(+1)

                SA_Score = self.eval(boxestemp, label_res)
                overall_SA.append(SA_Score)

                # --- build GT (T x 4) & exist ---
                annotemp = label_res['gt_rect']
                anno = np.array([a if len(a) == 4 else [0, 0, 0, 0] for a in annotemp], dtype=float)
                T = len(anno)
                exist = list(label_res.get('exist', [1] * T))
                if len(exist) < T:
                    exist += [0] * (T - len(exist))
                else:
                    exist = exist[:T]

                # --- build PRED (T x 4), pad with zeros (no truncation) ---
                boxes = np.array([b if len(b) == 4 else [0, 0, 0, 0] for b in boxestemp], dtype=float)
                boxes = self._pad_to_len(boxes, T)   # ← 길이 맞춤 (더미 패딩)
                # OTB 관례: 첫 프레임은 GT로 고정
                boxes[0] = anno[0]

                # metrics
                ious, center_errors = self._calc_metrics(boxes, anno)


                # valid mask: exist=1 & GT w,h>0
                valid = self._valid_mask(anno, exist)[:len(ious)]
                valid = valid.astype(bool)

                # # === DEBUG: 왼쪽 직선 원인(유효 프레임에서 IoU=0 비율) 확인 ===
                # num_valid = int(valid.sum())
                # num_total = int(len(valid))
                # num_exist = int(np.asarray(exist[:len(valid)], dtype=bool).sum())
                # if num_valid > 0:
                #     z0   = float((ious[valid] == 0).mean())      # IoU==0 비율
                #     z005 = float((ious[valid] < 0.05).mean())    # 아주 낮은 IoU 비율
                # else:
                #     z0 = z005 = float('nan')
                # print(f"[DBG] {name} | {seq_name} | valid={num_valid}/{num_total} "
                #       f"exist={num_exist}/{T} | zeroIoU={z0:.3f} | IoU<0.05={z005:.3f}")
                # # 필요시 심한 케이스 경고
                # if num_valid > 0 and z0 > 0.20:
                #     print(f"[WARN] {name} | {seq_name} zero-IoU ratio high ({z0:.2f}) → "
                #           f"왼쪽 급락 가능(Occlusion/OOV 재획득 실패 의심)")
                # # === DEBUG 끝 ===

                if valid.sum() == 0:
                    succ_curve[s] = np.zeros(self.nbins_iou, dtype=float)
                    prec_curve[s] = np.zeros(self.nbins_ce, dtype=float)
                else:
                    succ_curve[s], prec_curve[s] = self._calc_curves(ious[valid], center_errors[valid])


                # speed
                time_file = os.path.join(self.result_dir, name, 'times/%s_time.txt' % seq_name)
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.mean(1. / times)

                # per-sequence
                performance[name]['seq_wise'].update({
                    seq_name: {
                        'success_curve': succ_curve[s].tolist(),
                        'precision_curve': prec_curve[s].tolist(),
                        'success_score': float(np.mean(succ_curve[s])),
                        'precision_score': float(prec_curve[s][20]),
                        'success_rate': float(succ_curve[s][self.nbins_iou // 2]),
                        'speed_fps': float(speeds[s]) if speeds[s] > 0 else -1
                    }
                })

            overall_SAs.append(np.mean(overall_SA))

            # overall mean
            all_succ_curve = succ_curve.copy()
            all_prec_curve = prec_curve.copy()

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)

            succ_score = float(np.mean(succ_curve))
            prec_score = float(prec_curve[20])
            succ_rate = float(succ_curve[self.nbins_iou // 2])
            avg_speed = (np.sum(speeds) / np.count_nonzero(speeds)) if np.count_nonzero(speeds) > 0 else -1

            performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': float(avg_speed)
            })

            # attribute-wise
            att_array = np.array(att_list)
            for ii in range(len(self.att_name)):
                att_ids = np.where(att_array[:, ii] > 0)[0]
                if trackerid == 0:
                    self.att_name[ii] = self.att_name[ii] + '(' + str(len(att_ids)) + ')'
                att_succ_curve = np.mean(all_succ_curve[att_ids, :], axis=0) if len(att_ids) > 0 else np.zeros(self.nbins_iou)
                att_prec_curve = np.mean(all_prec_curve[att_ids, :], axis=0) if len(att_ids) > 0 else np.zeros(self.nbins_ce)
                performance[name][self.att_fig_name[ii]].update({
                    'att_success_curve': att_succ_curve.tolist(),
                    'att_precision_curve': att_prec_curve.tolist(),
                    'att_success_score': float(np.mean(att_succ_curve)),
                    'att_precision_score': float(att_prec_curve[20]),
                    'att_success_rate': float(att_succ_curve[self.nbins_iou // 2]),
                })

        # SA report
        sa_file = os.path.join(report_dir, 'state_accuracy_scores.txt')
        if os.path.exists(sa_file):
            os.remove(sa_file)
        text = '[Overall] %20s %20s: %s' % ('Tracker name', 'Experiments metric', 'Scores')
        with open(sa_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
        print(text)

        for ii in range(len(overall_SAs)):
            text = '[Overall] %20s %20s: %.04f' % (trackers[ii]['name'], 'State accuracy', overall_SAs[ii])
            with open(sa_file, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
            print(text)

        print('Saving state accuracy scores to', sa_file)
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        if plot_curves:
            self.plot_curves(tracker_names)
        if plot_attcurves:
            for ii in range(len(self.att_name)):
                self.plot_attcurves(tracker_names, self.att_name[ii], self.att_fig_name[ii])
        return performance

    # ---------- viz ----------
    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (s + 1, len(seq_names), seq_name))
            records = {}
            for name in tracker_names:
                record_file = os.path.join(self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    # ---------- IO ----------
    def _record(self, record_file, boxes, times, confs=None):
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        if confs is not None:
            lines = ['%.4f' % c for c in confs]
            lines[0] = ''
            conf_file = record_file.replace(".txt", "_confidence.value")
            with open(conf_file, 'w') as f:
                f.write(str.join('\n', lines))

        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    # ---------- metrics ----------
    def _calc_metrics(self, boxes, anno):
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce  = np.arange(0, self.nbins_ce)[np.newaxis, :]

        # IoU는 strict ‘>’로 (0에서 1.0로 시작하지 않음)
        bin_iou = np.greater(ious, thr_iou)
        # Precision은 보통 ‘<=’ 유지
        bin_ce  = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)
        return succ_curve, prec_curve


    def plot_curves(self, tracker_names):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        assert os.path.exists(report_dir), 'No reports found. Run "report" first before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), 'No reports found. Run "report" first before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.pdf')
        prec_file = os.path.join(report_dir, 'precision_plots.pdf')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        matplotlib.rcParams.update({'font.size': 6.8})

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][key]['success_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Overlap threshold', ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1.0), title='Success plots of OPE')
        ax.set_title('Success plots of OPE', fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving success plots to', succ_file)
        fig.savefig(succ_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)

        # precision curves (정렬은 일관성을 위해 성공 점수 기준 정렬을 그대로 써도 OK)
        matplotlib.rcParams.update({'font.size': 6.8})
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][key]['precision_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Location error threshold', ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1.0), title='Precision plots of OPE')
        ax.set_title('Precision plots of OPE', fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)

    def plot_attcurves(self, tracker_names, att_name, att_key):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        assert os.path.exists(report_dir), 'No reports found. Run "report" first before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), 'No reports found. Run "report" first before plotting curves.'

        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, f'success_plots_of_{att_key}.pdf')
        prec_file = os.path.join(report_dir, f'precision_plots_of_{att_key}.pdf')

        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        tracker_names = list(performance.keys())
        succ = [t[att_key]['att_success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        matplotlib.rcParams.update({'font.size': 6.8})

        # success (attribute)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][att_key]['att_success_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][att_key]['att_success_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][att_key]['att_success_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.98, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Overlap threshold', ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1.0), title='Success plots of OPE - ' + att_name)
        ax.set_title('Success plots of OPE - ' + att_name, fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving success plots to', succ_file)
        fig.savefig(succ_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)

        # precision (attribute)
        matplotlib.rcParams.update({'font.size': 6.8})
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines, legends = [], []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][att_key]['att_precision_curve'],
                            markers[i % len(markers)],
                            linewidth=2,
                            zorder=performance[name][att_key]['att_precision_score'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][att_key]['att_precision_score']))

        legend = ax.legend(lines, legends, bbox_to_anchor=(0.97, -0.19), loc="lower right",
                           bbox_transform=fig.transFigure, ncol=4, frameon=False)
        matplotlib.rcParams.update({'font.size': 9.0})
        ax.set(xlabel='Location error threshold', ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1.0), title='Precision plots of OPE - ' + att_name)
        ax.set_title('Precision plots of OPE - ' + att_name, fontweight='bold')
        ax.grid(True)
        fig.tight_layout()
        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=300)
