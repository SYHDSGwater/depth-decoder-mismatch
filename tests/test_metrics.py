import pytest
from ddm.metrics import head_regret, interaction_2x2, bits_per_byte, ols_slope

def test_metrics():
    assert head_regret(2.0,1.5)==0.5
    assert interaction_2x2(0.1,0.2,0.1,0.4) == pytest.approx(0.2)
    assert bits_per_byte(0.6931471805599453,1)==1.0
    assert ols_slope([1,2,3],[0.1,0.2,0.3]) > 0
