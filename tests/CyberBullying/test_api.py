import os
import pytest
from source.CyberBullying import api

ROOT = os.path.abspath(os.path.join(__file__, '../../../'))


def test_train():
    try:
        api.train(ROOT + '/source/CyberBullying/dataNew.csv')
    except Exception:
        pytest.fail("Unexpected error ..")


def test_get_classification_with_explain():
    classification = api.get_classification('אני אוהב אותך')
    assert 'class' in classification.keys()
    assert classification['class'] == 0
    assert 'explain' in classification.keys()
    assert type(classification['explain']) == str


def test_get_classification_without_explain():
    classification = api.get_classification('אני אוהב אותך', False)
    assert 'class' in classification.keys()
    assert classification['class'] == 0
    assert 'explain' not in classification.keys()


def test_get_performance():
    performance = api.get_performance(ROOT + '/source/CyberBullying/dataNew.csv')
    assert 'f-score' in performance.keys()
    assert 'recall' in performance.keys()
    assert 'precision' in performance.keys()
