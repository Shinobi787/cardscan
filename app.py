def extract_fields_from_text(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    data = {"Name": "", "Company": "", "Role": "", "Phone": "", "Email": "", "Website": ""}

    ROLE_KEYWORDS = [
        "founder", "co-founder", "director", "ceo", "cto", "cfo", "coo", "manager",
        "lead", "vp", "vice president", "president", "head", "engineer", "developer",
        "designer", "analyst", "consultant", "owner", "chairman", "marketing", "sales",
        "product", "operations", "hr", "support", "executive"
    ]

    COMPANY_SUFFIX = [
        "pvt", "private", "limited", "ltd", "llp", "inc", "llc", "company",
        "co.", "corporation", "enterprise", "technologies", "solutions",
        "studio", "labs", "global", "systems", "industries"
    ]

    # ------ EMAIL ------
    emails = EMAIL_REGEX.findall(text)
    if emails:
        data["Email"] = emails[0]

    # ------ PHONE ------
    all_phones = PHONE_REGEX.findall(text)
    cleaned = []
    for p in all_phones:
        p2 = re.sub(r"[^\d+]", "", p)
        if len(p2) >= 10:
            cleaned.append(p2)
    if cleaned:
        # Prefer mobile-length numbers (10-digit)
        mobiles = [x for x in cleaned if len(x.replace("+91", "")) == 10]
        data["Phone"] = mobiles[0] if mobiles else cleaned[0]

    # ------ WEBSITE ------
    webs = WEBSITE_REGEX.findall(text)
    if webs:
        for w in webs:
            if "." in w and not w.lower().startswith(("fax", "mob", "tel")):
                data["Website"] = w.strip()
                break

    # ------ NAME / ROLE / COMPANY ------
    # STEP 1: Pick the probable NAME (First reasonable line)
    for line in lines:
        clean = re.sub(r"[^a-zA-Z\s]", "", line).strip()

        # Name is usually 2–3 words capitalized
        if 2 <= len(clean.split()) <= 4:
            if not any(k in clean.lower() for k in ROLE_KEYWORDS):
                data["Name"] = clean.title()
                break

    # STEP 2: Find ROLE
    for line in lines:
        if any(k in line.lower() for k in ROLE_KEYWORDS):
            data["Role"] = line.strip().title()
            break

    # STEP 3: Find COMPANY
    for line in lines:
        low = line.lower()
        if any(s in low for s in COMPANY_SUFFIX):
            if len(line) < 50:  # avoid long address junk
                data["Company"] = line.strip().title()
                break

    # fallback if company empty → use prominent uppercase line
    if not data["Company"]:
        for line in lines:
            if line.isupper() and len(line.split()) <= 4:
                data["Company"] = line.title()
                break

    return data
