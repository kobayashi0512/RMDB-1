import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from diagnose_mass_operating_point import classify, margin, bias_dice_grid, label_metrics, pr_data

class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(33)
        self.p=r.uniform(.001,1,(4,24,24));self.p/=self.p.sum(0)
        self.t=r.integers(0,4,(24,24));self.t[0]=-100
    def test_argmax_and_bias(self):
        np.testing.assert_array_equal(classify(self.p),self.p.argmax(0))
        for b in [-3,-.7,0,.4,3]:
            logits=np.log(self.p);logits[1:]+=b
            np.testing.assert_array_equal(classify(self.p,b),logits.argmax(0))
    def test_bias_cannot_improve_margin_AP(self):
        y=self.t>0;s=margin(self.p)
        self.assertAlmostEqual(average_precision_score(y.ravel(),s.ravel()),average_precision_score(y.ravel(),(s+1.3).ravel()),12)
    def test_fast_dev_grid_matches_masks(self):
        bs=[-3,-1.4,0,.1,2.9]
        values=bias_dice_grid(self.p,self.t,bs)
        for v,b in zip(values,bs):
            d=label_metrics(classify(self.p,b),self.t,self.t>0,self.t>0,self.t>0,topology=False)
            self.assertAlmostEqual(v,d['mean_av_dice'],12)
    def test_oracle_precision_ties(self):
        y=np.array([1,0,1,1,0,0],bool);s=np.array([.9,.8,.8,.7,.2,.1])
        ap,curve,m=pr_data(y,s)
        self.assertAlmostEqual(ap,average_precision_score(y,s),12)
        for pt,found in m.items():
            if found:
                t,p,r=found;mask=s>=t
                self.assertGreaterEqual(p,pt)
                self.assertAlmostEqual(p,(y&mask).sum()/mask.sum())
                self.assertAlmostEqual(r,(y&mask).sum()/y.sum())
        self.assertTrue(np.all(np.diff(curve)<=1e-12))

if __name__=='__main__':unittest.main()
