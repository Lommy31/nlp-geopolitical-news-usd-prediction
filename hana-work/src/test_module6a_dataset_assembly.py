import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from module6a_dataset_assembly import BASE_FINANCIAL, assemble_dataset, fold_indices, load_calendar, load_metadata, load_fold_manifest


class Module6ATests(unittest.TestCase):
    def test_exact_fold_and_locked_membership(self):
        m=load_fold_manifest(); cal=load_calendar(); locked=set(m['locked_test']['calendar_indices'])
        for f in (1,2,3):
            train,val,lock=fold_indices(m,f); self.assertTrue(set(np.r_[train,val]).isdisjoint(locked)); self.assertEqual(cal.calendar_index.tolist(),list(range(1202))); self.assertEqual(set(lock),locked)

    def test_target_and_date_preservation(self):
        for mode in ('strict','same_date'):
            x=load_metadata(mode); self.assertTrue(x['target_date'].equals(x['next_Date'])); self.assertEqual(len(x),1202); self.assertIn('target_direction',x)

    def test_datasets_are_development_only_and_aligned(self):
        m=load_fold_manifest()
        for mode in ('strict','same_date'):
            d,info=assemble_dataset('B0',mode,1,m); self.assertTrue(set(d.split_role).issubset({'train','validation'})); self.assertTrue(d.calendar_index.is_unique); self.assertEqual(len(d),info['train_rows']+info['validation_rows']); self.assertTrue((d.Date<=d.target_date).all())

    def test_feature_dimensions_and_shared_financial_columns(self):
        m=load_fold_manifest()
        expected={'A2':12,'B0':15,'B1':30,'B2':20,'C1':115,'C2':783,'C3':783,'C4':783,'C5':783}
        for exp,dim in expected.items():
            d,_=assemble_dataset(exp,'strict',1,m,1 if exp in {'C4','C5'} else None); cols=[c for c in d.columns if c not in {'Date','calendar_index','split_role','target_date','next_Date','target_direction','target_available','is_flat_movement','is_warmup','supervised_eligible','module3_split','experiment_id','timing_mode','fold_id'}]; self.assertEqual(len(cols),dim)
            self.assertEqual(cols[:12],BASE_FINANCIAL)

    def test_a1_uses_only_previous_observed_direction(self):
        m=load_fold_manifest(); d,_=assemble_dataset('A1','strict',1,m); self.assertIn('last_observed_direction',d); self.assertFalse((d['last_observed_direction'].astype(str).str.contains('target_direction')).any())

    def test_manifest_and_output_metadata_after_run(self):
        p=Path('hana-work/data/manifests/module6a_dataset_manifest.json')
        if p.exists():
            d=json.loads(p.read_text()); self.assertEqual(d['dataset_count'],90); self.assertFalse(d['locked_test_rows_in_outputs']); self.assertFalse(d['preprocessing_fitted'])


if __name__=='__main__': unittest.main()
