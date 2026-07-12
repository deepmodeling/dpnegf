import os
from dpnegf.utils import argcheck


def _render_top_level(arg):
    """Render an Argument tree to RST using dargs' built-in generator.

    The `dargs` Argument tree already knows how to emit fully-nested RST
    (types, defaults, doc strings, and Variant branches). We just wrap the
    output in a header and a section anchor.
    """
    return arg.gen_doc(make_anchor=True, make_link=True)


def generate_rst_from_argcheck(output_dir="docs/input_params"):
    os.makedirs(output_dir, exist_ok=True)

    modules = {
        "common_options": argcheck.common_options,
        "run_options": argcheck.run_options,
    }

    for name, func in modules.items():
        arg = func()
        title = name.replace("_", " ").title()
        underline = "=" * len(title)
        body = _render_top_level(arg)
        parts = [f"{underline}\n{title}\n{underline}\n"]
        if name != arg.name:
            parts.append(f".. _`{name}`:\n")
        parts.append(body)
        rst = "\n".join(parts) + "\n"
        with open(os.path.join(output_dir, f"{name}.rst"), "w", encoding="utf-8") as f:
            f.write(rst)


def main():
    generate_rst_from_argcheck()


if __name__ == "__main__":
    main()
