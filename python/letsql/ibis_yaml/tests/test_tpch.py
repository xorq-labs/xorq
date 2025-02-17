import pytest


TPC_H = [
    "tpc_h01",
    "tpc_h02",
    "tpc_h03",
    "tpc_h04",
    "tpc_h05",
    "tpc_h06",
    "tpc_h07",
    "tpc_h08",
    "tpc_h09",
    "tpc_h10",
    "tpc_h11",
    "tpc_h12",
    "tpc_h13",
    "tpc_h14",
    "tpc_h15",
    "tpc_h16",
    "tpc_h17",
    "tpc_h18",
    "tpc_h19",
    "tpc_h20",
    "tpc_h21",
    "tpc_h22",
]


@pytest.mark.parametrize("fixture_name", TPC_H)
def test_yaml_roundtrip(fixture_name, compiler, request):
    query = request.getfixturevalue(fixture_name)

    yaml_dict = compiler.to_yaml(query)
    roundtrip_query = compiler.from_yaml(yaml_dict)

    assert roundtrip_query.equals(query), (
        f"Roundtrip expression for {fixture_name} does not match the original."
    )
