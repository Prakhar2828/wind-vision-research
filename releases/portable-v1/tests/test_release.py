import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
import replay

class ReleaseTests(unittest.TestCase):
    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'payload').write_bytes(b'original')
            (root/'MANIFEST.json').write_text(json.dumps({'files_sha256':{'payload':hashlib.sha256(b'original').hexdigest()}}))
            self.assertEqual(replay.verify(root),1)
            (root/'payload').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Integrity mismatch'): replay.verify(root)
    def test_manifest_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'MANIFEST.json').write_text(json.dumps({'files_sha256':{'../outside':'0'*64}}))
            with self.assertRaisesRegex(ValueError,'Unsafe manifest'): replay.verify(root)
    def test_constant_field_reproduced_without_hidden_values(self):
        xyz=np.array([[0,0,1],[1,0,1],[2,0,1],[3,0,1],[4,0,1]],dtype=float)
        obs=np.arange(4); hid=np.array([4]); values=np.full((2,4,16),7.25)
        for pred in replay.classical_predictions(xyz,values,obs,hid).values():
            np.testing.assert_allclose(pred,7.25,atol=1e-12,rtol=0)
        np.testing.assert_allclose(replay.transfer_weights(xyz,obs,hid).sum(axis=1),1)

if __name__=='__main__': unittest.main()
