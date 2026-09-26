"""Write ``Bundle``'s JSON schema (with every model in ``$defs``) for ``scripts/gen-contracts.sh``.

Usage: ``python -m hq_contracts.schema <out.json>``. Prints ``SCHEMA_VERSION`` on stdout.
"""

import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from hq_contracts.models import ALL_MODELS, SCHEMA_VERSION, Bundle


def _strip_property_titles(node: object) -> None:
    """Drop the per-field ``title`` pydantic adds (so json2ts names only models, not fields)
    and rewrite tuples into the draft-4 form json2ts understands."""
    if isinstance(node, dict):
        if "prefixItems" in node:
            # json2ts reads draft-4 tuple syntax (`items: [...]`), not 2020-12 `prefixItems`.
            node["items"] = node.pop("prefixItems")
            node["additionalItems"] = False
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for prop in value.values():
                    if isinstance(prop, dict):
                        prop.pop("title", None)
                        _strip_property_titles(prop)
            elif key != "$defs":
                _strip_property_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_property_titles(item)


def bundle_schema() -> dict:
    schema = Bundle.model_json_schema()
    defs = schema.setdefault("$defs", {})
    # Every model is reachable from Bundle today; keep this loop so an unreachable one still exports.
    for model in ALL_MODELS:
        if model is Bundle or model.__name__ in defs:
            continue
        sub = TypeAdapter(model).json_schema()
        defs.update(sub.pop("$defs", {}))
        defs[model.__name__] = sub
    for name, sub in defs.items():
        _strip_property_titles(sub)
        sub["title"] = name
    _strip_property_titles(schema)
    schema["title"] = "Bundle"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["description"] = f"Hidden Quakes contracts, schemaVersion {SCHEMA_VERSION}"
    return schema


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m hq_contracts.schema <out.json>", file=sys.stderr)
        return 2
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle_schema(), indent=2, sort_keys=True) + "\n")
    print(SCHEMA_VERSION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
