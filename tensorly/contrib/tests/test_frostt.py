from tensorly.contrib import frostt


def test_getting_frostt_dataset():
    x = frostt('lbnl-network')
    assert x.shape == (1605, 4198, 1631, 4209, 868131)
    assert 0 <= x.min() < x.max() <= 1460
