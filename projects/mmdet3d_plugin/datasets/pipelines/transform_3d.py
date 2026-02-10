import torch
import torchvision
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmcv.utils import build_from_cfg
from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter, ObjectNameFilter
from mmdet3d.core.bbox import CameraInstance3DBoxes, DepthInstance3DBoxes, LiDARInstance3DBoxes, box_np_ops
from projects.mmdet3d_plugin.datasets.builder import OBJECTSAMPLERS
from typing import Dict, Any
from PIL import Image

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus',
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'



@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str

@PIPELINES.register_module()
class ObjectRangeFilterTrack(object):
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        if 'gt_inds' in input_dict['ann_info'].keys():
            input_dict['gt_inds'] = input_dict['ann_info']['gt_inds']
        if 'gt_fut_traj' in input_dict['ann_info'].keys():
            input_dict['gt_fut_traj'] = input_dict['ann_info']['gt_fut_traj']
        if 'gt_fut_traj_mask' in input_dict['ann_info'].keys():
            input_dict['gt_fut_traj_mask'] = input_dict['ann_info']['gt_fut_traj_mask']
        if 'gt_past_traj' in input_dict['ann_info'].keys():
            input_dict['gt_past_traj'] = input_dict['ann_info']['gt_past_traj']
        if 'gt_past_traj_mask' in input_dict['ann_info'].keys():
            input_dict['gt_past_traj_mask'] = input_dict['ann_info']['gt_past_traj_mask']
        if 'gt_sdc_bbox' in input_dict['ann_info'].keys():
            input_dict['gt_sdc_bbox'] = input_dict['ann_info']['gt_sdc_bbox']
            input_dict['gt_sdc_label'] = input_dict['ann_info']['gt_sdc_label']
            input_dict['gt_sdc_fut_traj'] = input_dict['ann_info']['gt_sdc_fut_traj']
            input_dict['gt_sdc_fut_traj_mask'] = input_dict['ann_info']['gt_sdc_fut_traj_mask']

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_inds = input_dict['gt_inds']
        gt_fut_traj = input_dict['gt_fut_traj']
        gt_fut_traj_mask = input_dict['gt_fut_traj_mask']
        gt_past_traj = input_dict['gt_past_traj']
        gt_past_traj_mask = input_dict['gt_past_traj_mask']

        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        mask = mask.numpy().astype(bool)
        gt_labels_3d = gt_labels_3d[mask]
        gt_inds = gt_inds[mask]
        gt_fut_traj = gt_fut_traj[mask]
        gt_fut_traj_mask = gt_fut_traj_mask[mask]
        gt_past_traj = gt_past_traj[mask]
        gt_past_traj_mask = gt_past_traj_mask[mask]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_inds'] = gt_inds
        input_dict['gt_fut_traj'] = gt_fut_traj
        input_dict['gt_fut_traj_mask'] = gt_fut_traj_mask
        input_dict['gt_past_traj'] = gt_past_traj
        input_dict['gt_past_traj_mask'] = gt_past_traj_mask
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@PIPELINES.register_module()
class ObjectNameFilterTrack(object):
    """Filter GT objects by their names.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['gt_inds'] = input_dict['gt_inds'][gt_bboxes_mask]
        input_dict['gt_fut_traj'] = input_dict['gt_fut_traj'][gt_bboxes_mask]
        input_dict['gt_fut_traj_mask'] = input_dict['gt_fut_traj_mask'][gt_bboxes_mask]
        input_dict['gt_past_traj'] = input_dict['gt_past_traj'][gt_bboxes_mask]
        input_dict['gt_past_traj_mask'] = input_dict['gt_past_traj_mask'][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str

@PIPELINES.register_module()
class CustomObjectRangeFilter(ObjectRangeFilter):
    def __call__(self, results):
        """Call function to filter objects by the range.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(results['gt_bboxes_3d'],
                        (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(results['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        results['gt_bboxes_3d'] = gt_bboxes_3d
        results['gt_labels_3d'] = gt_labels_3d
        # results['ann_tokens'] = results['ann_tokens'][mask.numpy().astype(np.bool)]

        return results

@PIPELINES.register_module()
class CustomObjectNameFilter(ObjectNameFilter):
    def __call__(self, results):
        """Call function to filter objects by their names.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_labels_3d = results['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'][gt_bboxes_mask]
        results['gt_labels_3d'] = results['gt_labels_3d'][gt_bboxes_mask]
        # results['ann_tokens'] = results['ann_tokens'][gt_bboxes_mask]

        return results

@PIPELINES.register_module()
class ObjectPaste:
    """Sample GT objects to the data.
    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, stop_epoch=None):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.epoch = -1
        self.stop_epoch = stop_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, data):
        """Call function to sample ground truth objects to the data.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        # change to float for blending operation
        points = data["points"]
        if self.sample_2d:
            img = data["img"]
            gt_bboxes_2d = data["gt_bboxes"]
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img,
            )
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None
            )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict["gt_bboxes_2d"]
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]
                ).astype(np.float32)

                data["gt_bboxes"] = gt_bboxes_2d
                data["img"] = sampled_dict["img"]

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        data["points"] = points

        return data
    
@PIPELINES.register_module()
class GlobalRotScaleTrans_3D:
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = random.uniform(*self.resize_lim)
            theta = random.uniform(*self.rot_lim)
            translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            if "radar" in data:
                data["radar"].rotate(-theta)
                data["radar"].translate(translation)
                data["radar"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data

@PIPELINES.register_module()
class GTDepth:
    def __init__(self, keyframe_only=False):
        self.keyframe_only = keyframe_only 

    def __call__(self, data):
        sensor2ego = data['camera2ego'].data
        cam_intrinsic = data['camera_intrinsics'].data 
        img_aug_matrix = data['img_aug_matrix'].data 
        bev_aug_matrix = data['lidar_aug_matrix'].data
        lidar2ego = data['lidar2ego'].data 
        camera2lidar = data['camera2lidar'].data
        lidar2image = data['lidar2image'].data

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        points = data['points'].data 
        img = data['img'].data

        if self.keyframe_only:
            points = points[points[:, 4] == 0]

        batch_size = len(points)
        depth = torch.zeros(img.shape[0], *img.shape[-2:]) #.to(points[0].device)

        # for b in range(batch_size):
        cur_coords = points[:, :3]

        # inverse aug
        cur_coords -= bev_aug_matrix[:3, 3]
        cur_coords = torch.inverse(bev_aug_matrix[:3, :3]).matmul(
            cur_coords.transpose(1, 0)
        )
        # lidar2image
        cur_coords = lidar2image[:, :3, :3].matmul(cur_coords)
        cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = cur_coords[:, 2, :]
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = (
            (cur_coords[..., 0] < img.shape[2])
            & (cur_coords[..., 0] >= 0)
            & (cur_coords[..., 1] < img.shape[3])
            & (cur_coords[..., 1] >= 0)
        )
        for c in range(on_img.shape[0]):
            masked_coords = cur_coords[c, on_img[c]].long()
            masked_dist = dist[c, on_img[c]]
            depth[c, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        data['depths'] = depth 
        return data
    
@PIPELINES.register_module()
class GridMask:
    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results["img"]
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results
    
@PIPELINES.register_module()
class ImageNormalize:
    def __init__(self, mean, std, to_tensor=True):
        self.mean = mean
        self.std = std
        self.to_tensor = to_tensor
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = [self.compose(img) for img in data["img"]]
        if self.to_tensor:
            data["img"] = imgs
        else:
            data["img"] = [img.permute(1, 2, 0).cpu().numpy() for img in imgs]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data
    
@PIPELINES.register_module()
class BEVRandomFlip3D:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = random.choice([0, 1])
        flip_vertical = random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "radar" in data:
                data["radar"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("vertical")
            if "radar" in data:
                data["radar"].flip("vertical")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("vertical")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data

@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        ori_shape = results["ori_shape"]

        if isinstance(ori_shape, (list, tuple)) and isinstance(ori_shape[0], (list, tuple)):
            # multi-view: [(H, W, C), ...]
            H, W = ori_shape[0][:2]
        else:
            # single-view: (H, W, C)
            H, W = ori_shape[:2]

        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        if isinstance(img, np.ndarray):
            # img is HWC (uint8) usually
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            # some frameworks may wrap ndarray-like objects
            img = np.array(img)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)
            
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
        img = np.asarray(img)

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        for im in new_imgs:
            assert isinstance(im, np.ndarray), type(im)
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        return data
    
@PIPELINES.register_module()
class PointsToTensor:
    def __call__(self, results):
        if "points" in results:
            # LiDARPoints → torch.Tensor
            results["points"] = results["points"].tensor
        return results
