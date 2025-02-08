import ibis


def test_window_function_roundtrip(compiler, t):
    expr = t.select(
        [
            t.c.mean()
            .over(ibis.window(preceding=5, following=0, group_by=t.a))
            .name("mean_c")
        ]
    )

    yaml_dict = compiler.compile_to_yaml(expr)

    reconstructed_expr = compiler.compile_from_yaml(yaml_dict)

    assert expr.equals(reconstructed_expr)


def test_aggregation_window(compiler, t):
    cases = [
        (None, None),
        (5, 0),
        (0, 5),
        (5, 5),
    ]

    for preceding, following in cases:
        expr = t.select(
            [
                t.c.mean()
                .over(
                    ibis.window(preceding=preceding, following=following, group_by=t.a)
                )
                .name("mean_c")
            ]
        )

        yaml_dict = compiler.compile_to_yaml(expr)
        assert yaml_dict["op"] == "Project"
        window_func = yaml_dict["values"]["mean_c"]
        assert window_func["op"] == "WindowFunction"
        assert window_func["args"][0]["op"] == "Mean"

        if preceding is None:
            assert "start" not in window_func
        else:
            assert window_func["start"] == preceding

        if following is None:
            assert "end" not in window_func
        else:
            assert window_func["end"] == following

        print(yaml_dict)

        assert window_func["group_by"][0]["name"] == "a"
