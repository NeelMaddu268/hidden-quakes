"""Write ``Bundle``'s JSON schema (with every model in ``$defs``) for ``scripts/gen-contracts.sh``.

``schema.json`` exists only to generate the TS: it is normalized for json2ts (draft-07 tuples,
every field required). Validate JSON with the Pydantic models, never with this file.

Usage: ``python -m hq_contracts.schema <out.json>``. Prints ``SCHEMA_VERSION`` on stdout.
"""

import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from hq_contracts.models import ALL_MODELS, SCHEMA_VERSION, Bundle


def _strip_property_titles(node: object) -> None:
    """Normalize pydantic's schema for json2ts: drop per-field ``title`` (so only models get
    names), mark every field required (Python writes defaults too), and rewrite tuples into the
    draft-4 form json2ts understands."""
    if isinstance(node, dict):
        if "prefixItems" in node:
            # json2ts reads draft-4 tuple syntax (`items: [...]`), not 2020-12 `prefixItems`.
            node["items"] = node.pop("prefixItems")
            node["additionalItems"] = False
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                # Python serializes every field, defaults included, so every field is required.
                node["required"] = sorted(value)
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
    """JSON schema of everything, normalized so the generated TS has no optional fields."""
    schema = Bundle.model_json_schema(mode="serialization")
    defs = schema.setdefault("$defs", {})
    # Every model is reachable from Bundle today; keep this loop so an unreachable one still exports.
    for model in ALL_MODELS:
        if model is Bundle or model.__name__ in defs:
            continue
        sub = TypeAdapter(model).json_schema(mode="serialization")
        defs.update(sub.pop("$defs", {}))
        defs[model.__name__] = sub
    for name, sub in defs.items():
        _strip_property_titles(sub)
        sub["title"] = name
    _strip_property_titles(schema)
    schema["title"] = "Bundle"
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
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
