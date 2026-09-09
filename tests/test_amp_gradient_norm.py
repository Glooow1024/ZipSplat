"""A finite scaled gradient can overflow an FP32 norm, not the actual update."""
import math,unittest
import torch
from tools.scene_token.train_main import unscaled_gradient_norm

class AMPGradientNormTests(unittest.TestCase):
    def test_checkpoint_scale_reproduces_old_false_infinity(self):
        scale=2.**62;g=torch.tensor([5.,3.])*scale
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue(math.isinf(float(g.float().norm())/scale))
        self.assertAlmostEqual(unscaled_gradient_norm(g,scale),math.sqrt(34),places=12)

    def test_real_gradscaler_large_scale_update_stays_finite(self):
        p=torch.nn.Parameter(torch.tensor([1.,1.]));opt=torch.optim.SGD([p],lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=2.**62);scale=scaler.get_scale()
        scaler.scale((p*torch.tensor([5.,3.])).sum()).backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertTrue(torch.isinf(p.grad.norm()))
        measured=unscaled_gradient_norm(p.grad,scale)
        scaler.unscale_(opt)
        self.assertAlmostEqual(measured,float(p.grad.double().norm()),places=12)
        scaler.step(opt);scaler.update()
        torch.testing.assert_close(p,torch.tensor([.5,.7]))

    def test_zero_tiny_and_large_finite_values(self):
        self.assertEqual(unscaled_gradient_norm(torch.zeros(4),2.**62),0.)
        for magnitude in [1e-30,1.,1e30]:
            g=torch.full((1024,),magnitude)
            actual=unscaled_gradient_norm(g)
            self.assertTrue(math.isfinite(actual) and actual>0)
            self.assertAlmostEqual(actual/(float(g[0])*32),1.,places=12)

    def test_true_nonfinite_and_invalid_scale_still_fail(self):
        for bad in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(FloatingPointError):unscaled_gradient_norm(torch.tensor([bad]))
        for scale in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):unscaled_gradient_norm(torch.ones(1),scale)

if __name__=='__main__':unittest.main()
