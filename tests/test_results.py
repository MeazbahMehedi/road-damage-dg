import pytest

from rdd.evaluation.results import COLUMNS, append_result, read_results


def test_schema_and_duplicate(tmp_path):
    path = tmp_path / "results.csv"
    row = dict.fromkeys(COLUMNS, "")
    row.update(exp_id="e1", test_domain="Bangladesh", split="test", fold=0)
    with pytest.raises(ValueError, match="missing=.*map50"):
        append_result({k: v for k, v in row.items() if k != "map50"}, path)
    with pytest.raises(ValueError, match="unknown=.*wrong"):
        append_result(dict(row, wrong=1), path)
    append_result(row, path)
    with pytest.raises(ValueError, match="duplicate"):
        append_result(row, path)
    append_result(dict(row, map50=0.2), path, overwrite=True)
    append_result(dict(row, fold=1), path)
    assert [value["map50"] for value in read_results(path)] == ["0.2", ""]
