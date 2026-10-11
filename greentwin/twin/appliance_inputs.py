"""Simple appliance answers -> conditional centres, not measured consumption."""

import json
from pathlib import Path

import numpy as np


def appliance_centres(items, occupants):
    catalogue = json.loads(Path(__file__).with_name("household_catalogue.json").read_text())
    base, gas, shower, outdoor, secondary = (np.zeros_like(occupants) for _ in range(5))
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in catalogue["types"]:
            raise ValueError("Unknown appliance type; use the household catalogue")
        spec = catalogue["types"][item["type"]]
        answers = dict(item.get("answers", {}))
        for q in spec.get("questions", []):
            value = answers.get(q["key"], q["default"])
            if str(value) not in {str(option[0]) for option in q["options"]}:
                raise ValueError(f"Invalid appliance answer: {q['key']}")
            answers[q["key"]] = str(value)
        model = spec["model"]

        def lookup(table, default, answers=answers):
            for key, options in table.items():
                if answers.get(key) in options:
                    return float(options[answers[key]])
            return float(default)

        scale = occupants / catalogue["occupant_reference"] if model.get("scale_with_occupants") else 1
        kind = model["type"]
        if kind == "annual":
            annual = lookup(model.get("kwh_by", {}), model.get("base_kwh", 0))
        elif kind == "per_use":
            annual = lookup(model["kwh_per_use_by"], 1) * model["uses_per_week"] * 52.14
        elif kind == "power":
            annual = (
                lookup(model.get("watts_by", {}), model.get("base_watts", 0))
                * model["hours_per_day"]
                * 365
                / 1000
            )
        elif kind == "power_kw":
            annual = float(answers[model["kw_question"]]) * model["hours_per_day"] * 180
        elif kind == "shower":
            annual = float(answers["rating_kw"]) * model["minutes_per_person_per_day"] / 60 * occupants * 365
        elif kind == "ev":
            annual = float(answers["miles_per_week"]) * 52.14 * model["kwh_per_mile"]
        elif kind == "system":
            annual = 0
        else:
            raise ValueError("Unsupported appliance energy model")
        for key, options in model.get("multiplier_by", {}).items():
            annual *= options.get(answers.get(key), 1)
        annual = annual * scale
        category = spec["category"]
        fuel = "electricity"
        if "fuel_question" in spec:
            fuel = answers[spec["fuel_question"]]
        for key, options in spec.get("fuel_by", {}).items():
            fuel = options.get(answers.get(key), fuel)
        if fuel == "mains_gas" and "gas_multiplier" in spec:
            annual *= spec["gas_multiplier"]
        if category in ("appliance", "cooking"):
            if fuel == "electricity":
                base += annual / 365
            else:
                gas += annual / 365
        elif category == "hot_water":
            shower += annual / 365
        elif category == "outdoor":
            outdoor += annual / 365
        elif category == "secondary_heating":
            secondary += annual / 365
    return base, gas, shower, outdoor, secondary
