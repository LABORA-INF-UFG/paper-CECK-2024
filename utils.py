
import numpy as np
import cv2
import sys
import random
import colorsys
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

from glob import glob

def read_filter_classes(class_file_name):
    '''loads class name from a file'''
    with open(class_file_name, 'r') as file_object:
        str_classes = file_object.readline()
        return str_classes.split(',')




def im2single(I):
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.


def getWH(shape):
	return np.array(shape[1::-1]).astype(float)


def IOU(tl1,br1,tl2,br2):
	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())
	
	intersection_wh = np.maximum(np.minimum(br1,br2) - np.maximum(tl1,tl2),0.)
	intersection_area = np.prod(intersection_wh)
	area1,area2 = (np.prod(wh1),np.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area


def IOU_labels(l1,l2):
	return IOU(l1.tl(),l1.br(),l2.tl(),l2.br())


def IOU_centre_and_dims(cc1,wh1,cc2,wh2):
	return IOU(cc1-wh1/2.,cc1+wh1/2.,cc2-wh2/2.,cc2+wh2/2.)


def nms(Labels,iou_threshold=.5):

	SelectedLabels = []
	Labels.sort(key=lambda l: l.prob(),reverse=True)
	
	for label in Labels:

		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels(label,sel_label) > iou_threshold:
				non_overlap = False
				break

		if non_overlap:
			SelectedLabels.append(label)

	return SelectedLabels


def image_files_from_folder(folder,upper=True):
	extensions = ['jpg','jpeg','png']
	img_files  = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder,ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder,ext.upper()))
	return img_files


def is_inside(ltest,lref):
	return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()


# def crop_region(I,label,bg=0.5):
#
# 	wh = np.array(I.shape[1::-1])
#
# 	ch = I.shape[2] if len(I.shape) == 3 else 1
# 	tl = np.floor(label.tl()*wh).astype(int)
# 	br = np.ceil (label.br()*wh).astype(int)
# 	outwh = br-tl
#
# 	if np.prod(outwh) == 0.:
# 		return None
#
# 	outsize = (outwh[1],outwh[0],ch) if ch > 1 else (outwh[1],outwh[0])
# 	if (np.array(outsize) < 0).any():
# 		pause()
# 	Iout  = np.zeros(outsize,dtype=I.dtype) + bg
#
# 	offset 	= np.minimum(tl,0)*(-1)
# 	tl 		= np.maximum(tl,0)
# 	br 		= np.minimum(br,wh)
# 	wh 		= br - tl
#
# 	Iout[offset[1]:(offset[1] + wh[1]),offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1],tl[0]:br[0]]
#
# 	return Iout

def hsv_transform(I,hsv_modifier):
	I = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
	I = I + hsv_modifier
	return cv2.cvtColor(I,cv2.COLOR_HSV2BGR)


def bb_iou_2(tl1, br1, tl2, br2):
	boxA = [tl1[0], tl1[1], br1[0], br1[1]]
	boxB = [tl2[0], tl2[1], br2[0], br2[1]]
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou



def iou_alternative(cc1, wh1, cc2, wh2):
	return bb_iou_2(cc1-wh1/2.,cc1+wh1/2.,cc2-wh2/2.,cc2+wh2/2.)


def show(I,wname='Display'):
	cv2.imshow(wname, I)
	cv2.moveWindow(wname,0,0)
	key = cv2.waitKey(0) & 0xEFFFFF
	cv2.destroyWindow(wname)
	if key == 27:
		sys.exit()
	else:
		return key


ID_MOTO = 3


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def read_filter_classes(class_file_name):
    '''loads class name from a file'''
    with open(class_file_name, 'r') as file_object:
        str_classes = file_object.readline()
        return str_classes.split(',')



def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes




def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms_np_bboxes(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def carregar_cvat_ground_truth(caminho_arquivo_anotacoes):
	annotation_tree = ET.parse(caminho_arquivo_anotacoes)
	tag_raiz = annotation_tree.getroot()
	dict_bbox_gt = {}
	for indice_tag_image, image_tag in enumerate(tag_raiz.findall('image')):
		nome_frame = image_tag.get('name')
		boxes = image_tag.findall('box')
		lista_bbox_gt_frame = []
		for box in boxes:
			bbox_plate = [int(float(box.get('xtl'))), int(float(box.get('ytl'))), int(float(box.get('xbr'))), int(float(box.get('ybr'))), 0, 0, 0]
			top_left = float(box.get('xtl')), float(box.get('ytl'))
			bottom_right = float(box.get('xbr')), float(box.get('ybr'))
			lista_bbox_gt_frame.append((top_left[0],top_left[1], bottom_right[0], bottom_right[1]))
			# lista_bbox_gt_frame.append(bbox_plate)
			print(image_tag.tag, image_tag.attrib, box.get('label'))
		dict_bbox_gt[nome_frame] = np.array(lista_bbox_gt_frame)
	return dict_bbox_gt