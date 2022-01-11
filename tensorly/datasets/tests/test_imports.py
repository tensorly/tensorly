from ..imports import IL2data, COVID19_data


def test_IL2data():
    """ Test that data import dimensions match. """
    data = IL2data()

    tensor = data["tensor"]
    assert tensor.shape[0] == len(data["ligands"])
    assert tensor.shape[1] == len(data["times"])
    assert tensor.shape[2] == len(data["doses"])
    assert tensor.shape[3] == len(data["cells"])

def test_COVID19_data():
    """ Test that data import dimensions match. """
    data = COVID19_data()

    tensor = data["tensor"]
    assert tensor.shape[0] == len(data["samples"])
    assert tensor.shape[1] == len(data["antigens"])
    assert tensor.shape[2] == len(data["receptors"])

