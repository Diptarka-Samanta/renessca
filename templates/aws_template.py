aws_template = {
    "issuer": "Amazon Web Services",
    "keywords": ["Amazon Web Services", "AWS"],
    "fields": {
        "invoice_number": r"Invoice Number\s*:\s*(.*)",
        "invoice_date": r"Invoice Date\s*[:\-]?\s*(.*)",
        "due_date": r"TOTAL AMOUNT DUE ON\s+([A-Za-z]+\s+\d{1,2}\s*,\s*\d{4})",
        "total": r"TOTAL AMOUNT DUE ON\s+[A-Za-z]+\s+\d{1,2}\s*,\s*\d{4}\s+\$([0-9]+\.[0-9]{2})"
    }
}
