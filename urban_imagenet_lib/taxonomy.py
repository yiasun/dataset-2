"""HUSIC taxonomy utilities used by Task 1 and Task 2."""

from __future__ import annotations

HUSIC_CLASSES = [
    "Exterior urban spaces with people",
    "Exterior urban spaces without people",
    "Food or drink items",
    "Hotel or commercial lodging spaces",
    "Human-centered portrait",
    "Interior urban spaces with people",
    "Interior urban spaces without people",
    "Other non-spatial content",
    "Private home interiors",
    "Retail products and merchandise",
]

HUSIC_ID_TO_CLASS = dict(enumerate(HUSIC_CLASSES))
HUSIC_CLASS_TO_ID = {name: idx for idx, name in HUSIC_ID_TO_CLASS.items()}

HUSIC_PROMPTS = {
    "Exterior urban spaces with people": (
        "a social media photo of an activated exterior urban commercial space "
        "with visible pedestrians or people"
    ),
    "Exterior urban spaces without people": (
        "a social media photo of an exterior urban commercial space without "
        "visible people, emphasizing street, plaza, facade, or landscape design"
    ),
    "Food or drink items": "a social media photo centered on food, drinks, dessert, or dining items",
    "Hotel or commercial lodging spaces": "a social media photo of a hotel room or commercial lodging space",
    "Human-centered portrait": "a social media portrait or group photo where people are the main subject",
    "Interior urban spaces with people": (
        "a social media photo of an activated interior commercial space with "
        "visible shoppers, visitors, or workers"
    ),
    "Interior urban spaces without people": (
        "a social media photo of an interior commercial space without visible people"
    ),
    "Other non-spatial content": (
        "a social media image that is a screenshot, poster, advertisement, graphic, "
        "meme, or other non-spatial content"
    ),
    "Private home interiors": "a social media photo of a private home interior",
    "Retail products and merchandise": (
        "a social media photo centered on retail products, merchandise, fashion, cosmetics, or displays"
    ),
}


def prompt_for_class(class_name: str) -> str:
    """Return the CLIP-style category prompt for a HUSIC class."""

    return HUSIC_PROMPTS.get(class_name, f"a social media photo of {class_name}")
