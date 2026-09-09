"""Regression: real camera/pose collation must not recurse through tensordict."""
import unittest
import torch
from splatfactory.geometry import Camera, Pose
from splatfactory.datasets.utils.workers import collate


class TensorWrapperStackTests(unittest.TestCase):
    def test_camera_pose_stack_dimensions_and_gradients(self):
        for cls, width in [(Camera, 6), (Pose, 12)]:
            for dim in [0, 1, -1, -2]:
                x = torch.randn(3, width, requires_grad=True)
                result = torch.stack((cls(x), cls(x + 2)), dim=dim)
                expected = torch.stack([x, x + 2], dim=dim % 2)
                self.assertIsInstance(result, cls)
                torch.testing.assert_close(result.data_, expected)
                result.data_.sum().backward()
                torch.testing.assert_close(x.grad, torch.full_like(x, 2))
            with self.assertRaises(IndexError):
                torch.stack([cls(torch.zeros(3, width))], dim=2)

    def test_nested_batch_collation_and_out(self):
        sample = {'camera': Camera(torch.arange(12.).reshape(2, 6)),
                  'pose': Pose(torch.arange(24.).reshape(2, 12))}
        batch = collate([sample, sample])
        self.assertEqual(batch['camera'].data_.shape, (2, 2, 6))
        self.assertEqual(batch['pose'].data_.shape, (2, 2, 12))
        out = Camera(torch.empty(2, 2, 6))
        result = torch.stack([sample['camera'], sample['camera']], out=out)
        self.assertIs(result, out)
        torch.testing.assert_close(out.data_, batch['camera'].data_)


if __name__ == '__main__':
    unittest.main()
