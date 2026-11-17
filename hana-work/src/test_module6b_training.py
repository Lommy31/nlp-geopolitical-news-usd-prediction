import unittest
import numpy as np

from module6b_training import metrics, make_preprocessor, fold_indices, load_fold_manifest


class Module6BTests(unittest.TestCase):
    def test_metric_values_and_auc_guard(self):
        result=metrics(np.array([False,True,True,False]),np.array([False,True,False,False]),np.array([0.1,0.9,0.4,0.2]))
        self.assertAlmostEqual(result['mcc'], 0.577350269, places=6); self.assertIsNotNone(result['roc_auc']); self.assertEqual(set(result)&{'mcc','macro_f1','balanced_accuracy','accuracy','precision_down','precision_up','recall_down','recall_up'}, {'mcc','macro_f1','balanced_accuracy','accuracy','precision_down','precision_up','recall_down','recall_up'})

    def test_pca_dimension_rule(self):
        prep,dim,high=make_preprocessor([f'finbert_{i}' for i in range(768)],100); self.assertTrue(high); self.assertEqual(dim,64); self.assertEqual(prep['pca'].n_components,64)
        _,dim,high=make_preprocessor(['return_t']*12,100); self.assertFalse(high); self.assertEqual(dim,12)

    def test_fold_manifest_is_disjoint(self):
        m=load_fold_manifest()
        for f in (1,2,3):
            train,val,locked=fold_indices(m,f); self.assertFalse(train&val); self.assertFalse((train|val)&locked)


if __name__=='__main__': unittest.main()
