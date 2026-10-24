import pytest
from scripts.fpi_labels import parse_labels


def test_concatenated_boxes_are_recovered_and_exact_duplicates_removed():
    text='0 .4 .5 .1 .2 0 .4 .5 .1 .2\n1 .5 .3 .2 .3\n'
    assert parse_labels(text)==[(0,.4,.5,.1,.2),(1,.5,.3,.2,.3)]


@pytest.mark.parametrize('text',['0 .4 .5 .1','2 .4 .5 .1 .2','0 nan .5 .1 .2','0 .4 .5 0 .2'])
def test_malformed_source_annotations_are_rejected(text):
    with pytest.raises(ValueError):parse_labels(text)
