import unittest
import numpy as np
import pandas as pd
from three_readout_repeatability import estimate,eye_stats

class RepeatabilityTests(unittest.TestCase):
    def test_patient_copies_preserve_eyes(self):
        f=pd.DataFrame({'patient_id':['p1']*4+['p2']*2+['p3'],
                        'eye':['L','L','R','R','L','L','R']})
        for j,a in enumerate(['raw','mass','joint']):f[a+'_x']=np.array([1,3,5,6,2,4,90])+j
        r=eye_stats(f,'x');self.assertEqual(len(r),3)
        arrays=[np.stack([x[j] for x in r]) for j in [3,4,5,6]]
        e=estimate(np.array([[2.,2.,1.]]),np.array([x[2] for x in r]),*arrays)
        explicit=pd.concat([f[f.patient_id.eq('p1')].assign(patient_id='copy1'),
                            f[f.patient_id.eq('p1')].assign(patient_id='copy2'),f[f.patient_id.eq('p2')]])
        q=eye_stats(explicit,'x');other=estimate(np.ones((1,len(q))),np.array([x[2] for x in q]),
                                               *[np.stack([x[j] for x in q]) for j in [3,4,5,6]])
        for m in e:np.testing.assert_allclose(e[m],other[m],atol=1e-12)
    def test_singleton_and_missing_common_mask(self):
        f=pd.DataFrame({'patient_id':['p1','p1','p2','p2','p3'],'eye':['L']*5,
                        'raw_x':[1,2,3,4,9],'mass_x':[1,2,np.nan,4,9],'joint_x':[1,2,3,4,9]})
        r=eye_stats(f,'x');self.assertEqual(len(r),1);self.assertEqual(r[0][0],'p1')
    def test_balanced_icc(self):
        x=np.array([[1.,3.],[3.,7.],[8.,9.]])
        means=np.repeat(x.mean(1)[:,None],3,axis=1);ss=np.repeat(((x-x.mean(1)[:,None])**2).sum(1)[:,None],3,axis=1)
        mad=np.repeat(abs(x-x.mean(1)[:,None]).mean(1)[:,None],3,axis=1)
        e=estimate(np.ones((1,3)),np.full(3,2.),means,ss,mad,2*mad)
        B=2*((x.mean(1)-x.mean())**2).sum()/2;W=((x-x.mean(1)[:,None])**2).sum()/3
        np.testing.assert_allclose(e['icc'],(B-W)/(B+W))
        np.testing.assert_allclose(e['mean_pairwise_abs_diff'],2*e['within_eye_mad'])

if __name__=='__main__':unittest.main()
