from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
# from .data import cfg_mnet, cfg_re50
# from .layers.functions.prior_box import PriorBox
# from .utils.nms.py_cpu_nms import py_cpu_nms
# from .models.retinaface import RetinaFace
# from .utils.box_utils import decode, decode_landm
from anonymization_networks.Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from anonymization_networks.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from anonymization_networks.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from anonymization_networks.Pytorch_Retinaface.models.retinaface import RetinaFace
from anonymization_networks.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
import time
import cv2
# parser = argparse.ArgumentParser(description='Retinaface')
#
# parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
# args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class detect_face_in_video():
    def __init__(self, weight_file):
        self.cfg = cfg_re50
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, weight_file)
        self.net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cuda")
        self.normalization = torch.Tensor([104, 117, 123]).cuda()
        self.net = torch.nn.DataParallel(self.net.to(self.device, non_blocking=True), [i for i in range(torch.cuda.device_count())])
        self.resize = 1

        # im_dim[0] = height, im_dim[1] = width
        # testing begin

    def process(self, img, im_dim, postprocess, return_features=False):
        t = time.time()
        scale = torch.Tensor([im_dim[1], im_dim[0], im_dim[1], im_dim[0]])
        scale = scale.to(self.device, non_blocking=True)
        torch.set_grad_enabled(False)
        img -= self.normalization
        img = img.permute(0, 3, 1, 2)
        # img = img.to(self.device, non_blocking=True)
        # print("Preprocessing:", time.time() - t)
        t = time.time()
        with torch.no_grad():
            if return_features:
                (loc, conf, _), features = self.net(img, return_features)  # forward pass
            else:
                loc, conf, _ = self.net(img, return_features)
            # print("Processing:", time.time() - t)
            t = time.time()
            # import ipdb
            # ipdb.set_trace()
            priorbox = PriorBox(self.cfg, image_size=(int(im_dim[0]), int(im_dim[1])))  # im_dim has to be integers
            priors = priorbox.forward()
            priors = priors.to(self.device, non_blocking=True)
            prior_data = priors.data
            prior_data = prior_data.unsqueeze(0)
            prior_data = torch.cat([prior_data for _ in range(loc.shape[0])], 0)
            boxes = decode(loc.data, prior_data, self.cfg['variance'])
            boxes = boxes * scale / self.resize
            boxes = boxes.cpu().numpy()
        if postprocess:
            if return_features:
                return self.postprocess(boxes, conf), features[0]
            return self.postprocess(boxes, conf)
        else:
            if return_features:
                return boxes, conf, features[0]
            return boxes, conf

    def postprocess(self, boxes, conf):
        all_bboxes = []
        for i in range(boxes.shape[0]):
            scores = conf[i].squeeze(0).data.cpu().numpy()[:, 1]

            # ignore low scores
            vis_thres = 0.1
            inds = np.where(scores > vis_thres)[0]
            box = boxes[i][inds]
            scores = scores[inds]

            # keep top-K before NMS
            top_k = 5000
            order = scores.argsort()[::-1][:top_k]
            box = box[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((box, scores[:, np.newaxis])).astype(np.float32, copy=False)
            nms_threshold = 0.4
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep, :]

            # keep top-K faster NMS
            keep_top_k = 750
            all_bboxes.append(dets[:keep_top_k, :])
        # print("Postprocessing:", time.time() - t)
        return np.array(all_bboxes)



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    trained_model = './weights/Resnet50_Final.pth'
    cpu = False
    net = load_model(net, trained_model, cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(100):
        image_path = "./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        confidence_threshold = 0.02
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        top_k = 5000
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        nms_threshold = 0.4
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        keep_top_k = 750
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                vis_thres = 0.6
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)

