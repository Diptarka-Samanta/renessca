azure_template = {
    "issuer": "Azure Interior",
    "keywords": ["Azure Interior"],
    "fields": {
        "invoice_number": r"Invoice Number[:\s]*([A-Z0-9/-]+)",
        "invoice_date": r"Invoice Date\s+Due Date\s*\n([0-9]{2}/[0-9]{2}/[0-9]{4})",
        "due_date": r"Invoice Date\s+Due Date\s*\n[0-9]{2}/[0-9]{2}/[0-9]{4}\s+([0-9]{2}/[0-9]{2}/[0-9]{4})",
        "total": r"Total\s*\$([0-9]+\.[0-9]{2})"
    },

    "line_items": {
        "fields": {
            "item_code": r"\[([^\]]+)\]",
            "description": r"\]\s*(.*?)\s{2,}",
            "quantity": r"Qty\s*:\s*([0-9.]+)",
            "discount": r"Discount\s*:\s*([0-9.]+)",
            "tax_rate": r"Tax\s*:\s*([0-9.]+)",
            "amount": r"\$\s*([0-9.]+)$"
        }
    }
}
