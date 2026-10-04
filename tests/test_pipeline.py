"""Checks for segmentation geometry, odd U-Net sizes and mask AP edge cases."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from common import iou_arrays
from dataset import lane_mask, polygons, manifest
from evaluation import mask_average_precision
from model import CustomUNet
from train import loss_parts


class PipelineChecks(unittest.TestCase):
    def test_odd_dimensions_and_every_layer_gradient(self):
        torch.set_num_threads(2)
        model = CustomUNet()
        for h,w in [(36,64),(48,48),(37,65),(72,128)]:
            model.zero_grad(set_to_none=True)
            x = torch.rand(2,3,h,w)
            output = model(x)
            self.assertEqual(tuple(output.shape),(2,1,h,w))
            loss,_,_ = loss_parts(output,torch.rand_like(output).round())
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_selected_polygon_class_and_nearest_resize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'images/all_images').mkdir(parents=True)
            (root/'labels/all_images').mkdir(parents=True)
            image = root/'images/all_images/a.jpg'
            Image.new('RGB',(20,10)).save(image)
            label = root/'labels/all_images/a.txt'
            label.write_text('4 0 0 1 0 1 1 0 1\n3 0.25 0.2 0.75 0.2 0.75 0.8 0.25 0.8\n')
            mask = lane_mask(root,image,(20,10),names={3:'lane',4:'non-track area'})
            arr = np.array(mask)
            self.assertEqual(arr[0,0],0)
            self.assertEqual(arr[5,10],255)
            self.assertEqual(set(np.unique(np.array(mask.resize((7,3),Image.Resampling.NEAREST)))),{0,255})
            label.write_text('3 0 0 2 0 1 1\n')
            with self.assertRaises(ValueError):
                polygons(label,{3:'lane'})
            label.unlink()
            with self.assertRaises(FileNotFoundError):
                lane_mask(root,image,(20,10),names={3:'lane'})

    def test_empty_mask_loss_and_iou(self):
        empty = np.zeros((4,4),dtype=bool)
        self.assertEqual(iou_arrays(empty,empty),1.)
        self.assertEqual(iou_arrays(~empty,empty),0.)
        logits = torch.zeros(1,1,4,4,requires_grad=True)
        loss,_,_ = loss_parts(logits,torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mask_ap_false_positive_and_threshold_boundary(self):
        # Highest confidence candidate is a false positive; next matches 1 of 2 GTs.
        rows = [dict(ground_truth_foreground=False,predicted_foreground=True,confidence=.9,iou=0),
                dict(ground_truth_foreground=True,predicted_foreground=True,confidence=.8,iou=.8),
                dict(ground_truth_foreground=True,predicted_foreground=False,confidence=0,iou=0)]
        self.assertAlmostEqual(mask_average_precision(rows)['ap'],.25)
        boundary = [dict(ground_truth_foreground=True,predicted_foreground=True,confidence=.8,iou=.6)]
        self.assertEqual(mask_average_precision(boundary,.6)['ap'],0.)
        self.assertEqual(mask_average_precision(boundary,.6,True)['ap'],1.)
        self.assertIsNone(mask_average_precision(rows[:1])['ap'])

    def test_ap_ties_grouped_without_order_bias(self):
        rows = [dict(ground_truth_foreground=False,predicted_foreground=True,confidence=.8,iou=0),
                dict(ground_truth_foreground=True,predicted_foreground=True,confidence=.8,iou=.8)]
        self.assertEqual(mask_average_precision(rows)['ap'],.5)
        self.assertEqual(mask_average_precision(list(reversed(rows)))['ap'],.5)


if __name__=='__main__':
    unittest.main()
