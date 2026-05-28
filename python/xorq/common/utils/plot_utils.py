from xorq.expr import udf


PLOT_TAG = "plot"


def make_plot_expr(table, plot_fn, return_type, *, name="plot"):
    plot_udaf = udf.agg.pandas_df(
        plot_fn,
        table.schema(),
        return_type,
        name=name,
    )
    return table.agg(plot_udaf.on_expr(table).name(name)).tag(PLOT_TAG)
