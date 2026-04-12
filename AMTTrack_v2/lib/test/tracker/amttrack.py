import math

import numpy as np
from lib.models.amttrack import build_amttrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import torch.nn.functional as F
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond, generate_mask_z
from lib.models.layers.thor import THOR_Wrapper
from lib.test.tracker.motion import KalmanMotionModel


class AMTTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(AMTTrack, self).__init__(params)
        network = build_amttrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
    
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes 
        self.thor_wrapper = THOR_Wrapper(net=self.network, 
                                         st_capacity=params.cfg.TEST.SHORTTERM_LIBRARY_NUMS, 
                                         lt_capacity=params.cfg.TEST.LONGTERM_LIBRARY_NUMS,
                                         sample_interval=params.cfg.TEST.SAMPLE_INTERVAL, 
                                         update_interval=params.cfg.TEST.UPDATE_INTERVAL, 
                                         lower_bound=params.cfg.TEST.LOWER_BOUND,
                                         score_threshold=params.cfg.TEST.SCORE_THRESHOLD,
                                         )

    def initialize(self, image, event_image, info: dict, idx=0):
        # forward the template once
        z_patch_arr, event_z_patch_arr, resize_factor, z_amask_arr = sample_target(im=image, eim=event_image,
            target_bb=info['init_bbox'], search_area_factor=self.params.template_factor, output_sz=self.params.template_size)

        self.z_patch_arr = z_patch_arr
        self.event_z_patch_arr = event_z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
        event_template = self.preprocessor.process(event_z_patch_arr, z_amask_arr).tensors

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor, template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
        self.mask_z = generate_mask_z(cfg=self.cfg, bs=1, device=template.device, gt_bbox=template_bbox)

        with torch.no_grad():
            self.dynamic_zi = None
            self.dynamic_ze = None
            self.static_zi = template  # (1, 3, 128, 128)
            self.static_ze = event_template  # (1, 3, 128, 128)
            self.thor_wrapper.setup(self.static_zi, self.static_ze)
        
        self.state = info['init_bbox']
        self.frame_id = idx

        # motion model — initialised here so it resets cleanly per sequence
        self.use_motion_model = getattr(self.cfg.TEST, 'MOTION_MODEL', False)
        self.motion_model = KalmanMotionModel()
        self.motion_model.update(self.state)

        # blind period diagnostics
        self.blind_frames_count = 0    # consecutive frames below score threshold (diagnostic)
        self.genuine_blind_count = 0   # consecutive frames where score crashed vs EMA (drives Kalman)
        self.score_ema = None          # reset rolling average per sequence
        self.event_density_ema = None  # reset event density baseline per sequence


    def track(self, image, event_image, info: dict = None):
        self.frame_id += 1
        H, W, _ = image.shape

        # genuine_blind_count: counts consecutive frames where the score genuinely
        # crashed relative to its rolling average — drives both Kalman and search widening.
        # Resets as soon as the score recovers above the DROP_RATIO threshold.
        # This is distinct from blind_frames_count (raw sub-threshold counter for diagnostics)
        # which never resets on sequences with uniformly low scores.
        prev_score = (info['previous_output'].get('pred_score', 1.0)
                      if info and info.get('previous_output') else 1.0)
        if self.score_ema is None:
            self.score_ema = prev_score
        self.score_ema = 0.95 * self.score_ema + 0.05 * prev_score
        prev_genuine_blind = (info['previous_output'].get('genuine_blind_frames', 0)
                              if info and info.get('previous_output') else 0)
        is_crash = prev_score < self.score_ema * self.cfg.TEST.KALMAN_DROP_RATIO
        genuine_blind_count = prev_genuine_blind + 1 if is_crash else 0

        blind = (self.use_motion_model
                 and self.cfg.TEST.KALMAN_MIN_BLIND <= genuine_blind_count <= self.cfg.TEST.KALMAN_MAX_BLIND)

        # Adaptive search widening: only when genuinely crashed past the Kalman window
        if self.use_motion_model and genuine_blind_count > self.cfg.TEST.KALMAN_MAX_BLIND:
            frames_past_max = genuine_blind_count - self.cfg.TEST.KALMAN_MAX_BLIND
            scale = min(
                1.0 + (frames_past_max / self.cfg.TEST.KALMAN_MAX_BLIND)
                      * (self.cfg.TEST.KALMAN_SEARCH_SCALE_MAX - 1.0),
                self.cfg.TEST.KALMAN_SEARCH_SCALE_MAX
            )
            effective_search_factor = self.params.search_factor * scale
        else:
            effective_search_factor = self.params.search_factor

        x_patch_arr, event_x_patch_arr, resize_factor, x_amask_arr = sample_target(im=image, eim=event_image,
            target_bb=self.state, search_area_factor=effective_search_factor, output_sz=self.params.search_size)   # (x1, y1, w, h)

        # --- event density spike detection (diagnostic) ---
        # Mean absolute intensity of the raw event crop (uint8, [0,255]) before any normalization
        event_density_raw = float(np.abs(event_x_patch_arr.astype(np.float32)).mean())
        EMA_ALPHA = getattr(self.cfg.TEST, 'DENSITY_EMA_ALPHA', 0.05)
        SPIKE_RATIO = getattr(self.cfg.TEST, 'SPIKE_RATIO', 2.0)
        prev_ema = self.event_density_ema
        if self.event_density_ema is None:
            self.event_density_ema = event_density_raw
        else:
            self.event_density_ema = (EMA_ALPHA * event_density_raw
                                      + (1 - EMA_ALPHA) * self.event_density_ema)
        # compare against prev_ema (before this frame contaminated the baseline)
        event_spike = (prev_ema is not None
                       and event_density_raw > SPIKE_RATIO * prev_ema
                       and prev_ema > 0)
        # --- end event density ---

        search = self.preprocessor.process(x_patch_arr, x_amask_arr).tensors
        event_search = self.preprocessor.process(event_x_patch_arr, x_amask_arr).tensors

        with torch.no_grad():
            if len(info['previous_output']) == 0:   
                self.dynamic_zi, self.dynamic_ze = self.thor_wrapper.update(self.static_zi, self.static_ze, 1)
            else:
                self.dynamic_zi, self.dynamic_ze = self.thor_wrapper.update(
                                                        info['previous_output']['prediction_image'],
                                                        info['previous_output']['prediction_event_image'], 
                                                        info['previous_output']['pred_score'])
            out_dict = self.network.inference(
                static_zi=self.static_zi, static_ze=self.static_ze, 
                dynamic_zi=self.dynamic_zi, dynamic_ze=self.dynamic_ze, 
                xi=search, xe=event_search)
            
        # add hann windows
        pred_score_map = out_dict['score_map']  
        response = self.output_window * pred_score_map 
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map']) 
        pred_boxes = pred_boxes.view(-1, 4) 
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # blind period counter — tracks consecutive low-confidence frames
        current_score = response.max().item()
        if current_score < self.cfg.TEST.SCORE_THRESHOLD:
            self.blind_frames_count += 1
        else:
            self.blind_frames_count = 0

        # Hook B — motion model injection
        if self.use_motion_model:
            if blind:
                # network output is unreliable: replace with Kalman prediction
                self.state = clip_box(self.motion_model.predict_only(), H, W, margin=10)
            else:
                # network output is reliable: feed it to the filter as observation
                self.motion_model.update(self.state)

        ##################################################################################
        # extract the tracking result
        tracking_result_arr, tracking_result_event_arr, resize_factor, tracking_result_amask_arr = sample_target(
            im=image, eim=event_image, target_bb=self.state, search_area_factor=self.params.template_factor, output_sz=self.params.template_size)
        prediction_image = self.preprocessor.process(tracking_result_arr, tracking_result_amask_arr).tensors
        prediction_event_image = self.preprocessor.process(tracking_result_event_arr, tracking_result_amask_arr).tensors
        ##################################################################################

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        return {"target_bbox": self.state,
                "prediction_image": prediction_image,
                'prediction_event_image': prediction_event_image,
                "response": response,
                "pred_score": current_score,
                "blind_frames": self.blind_frames_count,
                "genuine_blind_frames": genuine_blind_count,
                "event_density": event_density_raw,
                "event_density_ema": self.event_density_ema,
                "event_spike": int(event_spike),
                }
   
    def get_count(self):
        st_count, lt_count = self.thor_wrapper.get_count()
        return st_count, lt_count

    def get_update_count(self):
        st_update_count, lt_update_count, frame_count = self.thor_wrapper.get_update_count()
        return st_update_count, lt_update_count, frame_count
    
    def get_sample_count(self):
        sample_count, frame_count = self.thor_wrapper.get_sample_count()
        return sample_count, frame_count
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )
        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return AMTTrack
