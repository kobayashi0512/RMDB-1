import unittest
import numpy as np
from final_measurement_controls import simple_readouts, binary_mask, retention, reference_features

class ControlsTest(unittest.TestCase):
    def test_cleanup_component_area_and_identity(self):
        p=np.zeros((4,64,64),np.float32);p[0]=1
        p[:,10:14,10:18]=0;p[1,10:14,10:18]=.9;p[0,10:14,10:18]=.1 #32 retained
        p[:,30:33,30:40]=0;p[2,30:33,30:40]=.8;p[0,30:33,30:40]=.2 #30 removed
        q=simple_readouts(p,np.ones((64,64),bool))['cleanup']
        np.testing.assert_array_equal(q[:,10:14,10:18],p[:,10:14,10:18])
        self.assertTrue((q[0,30:33,30:40]==1).all());self.assertTrue((q[1:,30:33,30:40]==0).all())
        np.testing.assert_allclose(q.sum(0),1.)
    def test_smoothing_does_not_mix_channels(self):
        p=np.zeros((4,64,64),np.float32);p[0]=.2;p[1]=.8
        q=simple_readouts(p,np.ones((64,64),bool))['smooth']
        np.testing.assert_allclose(q,p,atol=1e-7)
    def test_perfect_reference(self):
        t=np.zeros((512,512),bool);t[100:105,30:480]=True;t[30:400,100:105]=True
        r=retention(t,t)
        for k,v in r.items():self.assertAlmostEqual(v,1.)
        f=reference_features(t,np.ones_like(t));self.assertAlmostEqual(f['vessel_density'],t.mean())
    def test_empty_prediction_fails(self):
        t=np.zeros((512,512),bool);t[100:102,50:400]=True
        r=retention(np.zeros_like(t),t)
        for k,v in r.items():self.assertEqual(v,0.)

if __name__=='__main__':unittest.main()
