"""Extraction schemas: main entity type + typed slots with definitions, localized labels and aliases."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import lexicon as L


@dataclass
class Slot:
    name: str
    type: str  # id|date|location|org|status|person|amount|text
    definition: str
    gen: str  # lexicon generator key or "date"/"status"
    aliases: list[str] = field(default_factory=list)
    labels: dict = field(default_factory=dict)


@dataclass
class Schema:
    id: str
    main_type: str
    main_definition: str
    slots: list[Slot]
    container_slot: str  # slot holding the containing org (H6)
    id_slot: str | None = None

    def slot(self, name: str) -> Slot:
        return next(s for s in self.slots if s.name == name)

    def to_prompt_dict(self, rename: dict | None = None, drop_defs: bool = False) -> dict:
        rename = rename or {}
        return {
            "main_entity": {"type": self.main_type, **({} if drop_defs else {"definition": self.main_definition})},
            "slots": [{"name": rename.get(s.name, s.name), "type": s.type,
                       **({} if drop_defs else {"definition": s.definition})} for s in self.slots],
        }


def _vehicle_main(rng, lang):
    v, c = rng.choice(L.VEHICLES[lang]), rng.choice(L.COLORS[lang])
    return {"he": f"{v} {c}", "fa": f"{v} {c}", "en": f"{c} {v}"}[lang] if rng.random() < 0.7 else v


MAIN_GEN = {
    "vehicle": _vehicle_main,
    "shipment": L.shipment_main,
    "person_employment": L.person,
    "asset": L.asset_main,
    "meeting": L.meeting_main,
    "contract": L.contract_main,
    "account": L.account_main,
    "vessel": L.vessel_main,
}

SCHEMAS: dict[str, Schema] = {s.id: s for s in [
    Schema("vehicle", "vehicle", "A specific vehicle (make/model, optionally color) mentioned in the document.", [
        Slot("plate", "id", "License plate number of the vehicle.", "plate", ["license_plate", "registration"]),
        Slot("location", "location", "Where the vehicle currently is / is parked (not where it was made or previously was).",
             "location", ["current_location", "parked_at"]),
        Slot("date", "date", "The date of the main upcoming/current event for the vehicle (pick-up, arrival, release). "
             "Not manufacture dates or dates of earlier messages.", "date", ["event_date", "pickup_date"]),
        Slot("status", "status", "Latest status of the vehicle.", "status", ["current_status", "state"]),
        Slot("owner_org", "org", "Organization that owns / operates the vehicle (not the manufacturer or insurer).",
             "org", ["owner", "fleet_owner"]),
    ], "owner_org", "plate"),
    Schema("shipment", "shipment", "A shipment / consignment of goods.", [
        Slot("tracking_id", "id", "Tracking number of the shipment.", "tracking_id", ["tracking_number", "awb"]),
        Slot("origin", "location", "Where the shipment was sent from.", "location", ["from"]),
        Slot("destination", "location", "Where the shipment is going to.", "location", ["to", "dest"]),
        Slot("date", "date", "Expected or actual delivery date.", "date", ["delivery_date", "eta"]),
        Slot("status", "status", "Latest status of the shipment.", "status", ["state"]),
        Slot("carrier_org", "org", "Carrier/logistics company handling the shipment.", "org", ["carrier"]),
    ], "carrier_org", "tracking_id"),
    Schema("person_employment", "person", "A person and their current employment.", [
        Slot("id_number", "id", "Personal / employee identification number.", "national_id", ["person_id", "employee_id"]),
        Slot("role", "text", "Current job title / role.", "role", ["job_title", "position"]),
        Slot("employer_org", "org", "Current employer (not former employers).", "org", ["employer", "company"]),
        Slot("location", "location", "Where the person currently works.", "location", ["work_location"]),
        Slot("start_date", "date", "Start date of the current job (not birth date, not previous jobs).", "date",
             ["employment_start", "hired_on"]),
    ], "employer_org", "id_number"),
    Schema("asset", "asset", "An equipment item / asset.", [
        Slot("serial", "id", "Serial number of the asset.", "serial", ["serial_number", "sn"]),
        Slot("assigned_to", "person", "Person currently holding the asset.", "person", ["holder", "custodian"]),
        Slot("location", "location", "Current location of the asset.", "location", ["site"]),
        Slot("date", "date", "Date of current assignment / return.", "date", ["assignment_date"]),
        Slot("status", "status", "Latest status of the asset.", "status", ["state"]),
        Slot("owner_org", "org", "Organization that owns the asset.", "org", ["owner"]),
    ], "owner_org", "serial"),
    Schema("meeting", "event", "A meeting / event.", [
        Slot("date", "date", "The date the event takes place (latest scheduled date).", "date", ["event_date", "when"]),
        Slot("location", "location", "Where the event takes place.", "location", ["venue", "where"]),
        Slot("organizer_org", "org", "Organization that organizes the event.", "org", ["organizer", "host"]),
        Slot("status", "status", "Latest status of the event.", "status", ["state"]),
    ], "organizer_org", None),
    Schema("contract", "contract", "A contract / agreement.", [
        Slot("contract_id", "id", "Contract number.", "contract_id", ["agreement_no", "ref"]),
        Slot("supplier_org", "org", "Supplier organization.", "org", ["supplier", "vendor"]),
        Slot("client_org", "org", "Client organization.", "org", ["client", "buyer"]),
        Slot("date", "date", "Signing / effective date.", "date", ["signed_on", "effective_date"]),
        Slot("value", "amount", "Monetary value of the contract.", "amount", ["amount", "price"]),
        Slot("status", "status", "Latest status of the contract.", "status", ["state"]),
    ], "client_org", "contract_id"),
    Schema("account", "bank_account", "A bank account.", [
        Slot("account_number", "id", "Bank account number.", "account_number", ["acct_no", "iban"]),
        Slot("holder", "person", "Account holder.", "person", ["owner_name"]),
        Slot("bank_org", "org", "The bank where the account is held.", "bank", ["bank"]),
        Slot("branch_location", "location", "Branch location.", "location", ["branch"]),
        Slot("date", "date", "Date of the latest account action (opening/freeze/closing).", "date", ["action_date"]),
        Slot("status", "status", "Latest account status.", "status", ["state"]),
    ], "bank_org", "account_number"),
    Schema("vessel", "vessel", "A ship / vessel.", [
        Slot("imo", "id", "IMO number of the vessel.", "imo", ["imo_number", "ship_id"]),
        Slot("port", "location", "Port where the vessel currently is or is heading to.", "location", ["current_port"]),
        Slot("date", "date", "Arrival / departure date of the current voyage.", "date", ["eta", "voyage_date"]),
        Slot("status", "status", "Latest status of the vessel.", "status", ["state"]),
        Slot("operator_org", "org", "Operator / owner company of the vessel.", "org", ["operator", "owner"]),
    ], "operator_org", "imo"),
]}


def gen_value(schema: Schema, slot: Slot, rng: random.Random, lang: str) -> dict:
    """Return {'value': surface, 'iso': iso-date|None}."""
    if slot.gen == "date":
        d = L.random_date(rng)
        s, style = L.format_date(d, lang, rng)
        return {"value": s, "iso": None if style == "hebcal" else d.isoformat(), "style": style}
    if slot.gen == "status":
        return {"value": rng.choice(L.STATUSES[schema.id if schema.id in L.STATUSES else "asset"][lang])}
    return {"value": L.GENERATORS[slot.gen](rng, lang)}


def gen_main(schema: Schema, rng: random.Random, lang: str) -> str:
    return MAIN_GEN[schema.id](rng, lang)


def schema_from_dict(sid: str, d: dict) -> Schema:
    """Inline schema (e.g. converted public data): {"main_type", "main_definition", "slots": [{name,type,definition}]}."""
    slots = [Slot(s["name"], s.get("type", "text"), s.get("definition", ""), "", s.get("aliases", []))
             for s in d["slots"]]
    return Schema(sid, d.get("main_type", "entity"), d.get("main_definition", ""), slots,
                  d.get("container_slot", ""), d.get("id_slot"))


def schema_of(sample: dict) -> Schema:
    if sample.get("schema"):
        return schema_from_dict(sample["schema_id"], sample["schema"])
    return SCHEMAS[sample["schema_id"]]
