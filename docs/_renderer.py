from __future__ import annotations

from textwrap import dedent

import quartodoc as qd
import toolz
from plum import dispatch


class Renderer(qd.MdRenderer):
    style = "ibis"

    @dispatch
    def render(self, el: qd.ast.ExampleCode) -> str:
        doc = el.value.replace('x.cast("uint16")', 'x.cast("int8")')
        lines = doc.splitlines()

        result = []

        prompt = ">>> "
        continuation = "..."

        skip_doctest = "doctest: +SKIP"
        expect_failure = "quartodoc: +EXPECTED_FAILURE"
        quartodoc_skip_doctest = "quartodoc: +SKIP"

        def chunker(line):
            return line.startswith((prompt, continuation))

        def should_skip(line):
            return quartodoc_skip_doctest in line or skip_doctest in line

        for chunk in toolz.partitionby(chunker, lines):
            first, *rest = chunk

            # only attempt to execute or render code blocks that start with the
            # >>> prompt
            if first.startswith(prompt):
                # check whether to skip execution and if so, render the code
                # block as `python` (not `{python}`) if it's marked with
                # skip_doctest, expect_failure or quartodoc_skip_doctest
                if skipped := any(map(should_skip, chunk)):
                    start = end = ""
                else:
                    start, end = "{}"
                    result.append(
                        dedent(
                            """
                            ```{python}
                            #| echo: false
                            
                            import xorq.api as xo
                            xo.options.interactive = True
                            ```
                            """
                        )
                    )

                result.append(f"```{start}python{end}")

                # if we expect failures, don't fail the notebook execution and
                # render the error message
                if expect_failure in first or any(
                    expect_failure in line for line in rest
                ):
                    assert start and end, (
                        "expected failure should never occur alongside a skipped doctest example"
                    )
                    result.append("#| error: true")

                # remove the quartodoc markers from the rendered code
                result.append(
                    first.replace(f"# {quartodoc_skip_doctest}", "")
                    .replace(quartodoc_skip_doctest, "")
                    .replace(f"# {expect_failure}", "")
                    .replace(expect_failure, "")
                )
                result.extend(rest)
                result.append("```\n")

                if not skipped:
                    result.append(
                        dedent(
                            """
                            ```{python}
                            #| echo: false
                            xo.options.interactive = False
                            ```
                            """
                        )
                    )

        return "\n".join(result)
