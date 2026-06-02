"""
Location term expansion for the "Find Similar Profiles" location filter.

The user types one thing — a country picked from a dropdown ("Israel") or a
city typed free-hand ("Tel Aviv") — and we quietly expand it into every
related term we should match against the free-text ``location`` field on a
profile ("Tel Aviv, Israel", "Greater Tel Aviv Area", "Herzliya", ...).

Why this exists:
    Profiles store location as one free-text string the person wrote on
    LinkedIn. Some say "Israel", some only say "Tel Aviv", some say
    "Ramat Gan". A naive "contains Israel" filter misses the last two. So we
    map each country to its cities and each known city to its variations, and
    match if the profile location contains ANY of the expanded terms.

Coverage (by design):
    - Israel is covered exhaustively (every meaningful city/area).
    - Major global tech hubs are covered well.
    - Any city NOT in the map still works — it just matches the literal text
      the user typed, with no extra synonyms. Nothing breaks; it's only
      "less smart" for places we haven't curated.

All terms are lowercase substrings, intended for case-insensitive
"contains" (ILIKE ``*term*``) matching — the same style the rest of the app
uses for the location text box.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# City → variations / synonyms / immediate metro.
#
# Keyed by a normalized city name (lowercase). Looking up "tel aviv" should
# also catch "tel aviv-yafo", "jaffa", "greater tel aviv area", etc. Keep the
# values tight to the city and its immediate metro — this is "what counts as
# the same place", not "everything in the country".
# ---------------------------------------------------------------------------
CITY_TERMS: dict[str, list[str]] = {
    # --- Israel ---------------------------------------------------------
    "tel aviv": [
        "tel aviv", "tel aviv-yafo", "tel aviv yafo", "tel-aviv",
        "jaffa", "yafo", "greater tel aviv", "gush dan", "tel aviv district",
    ],
    "ramat gan": ["ramat gan", "ramat-gan"],
    "givatayim": ["givatayim", "giv'atayim"],
    "herzliya": ["herzliya", "herzeliya", "herzlia"],
    "petah tikva": ["petah tikva", "petach tikva", "petah tiqwa", "petah-tikva"],
    "rishon lezion": ["rishon lezion", "rishon le zion", "rishon le-zion", "rishon"],
    "holon": ["holon"],
    "bat yam": ["bat yam", "bat-yam"],
    "bnei brak": ["bnei brak", "bene beraq", "bnei-brak"],
    "ramat hasharon": ["ramat hasharon", "ramat ha'sharon", "ramat-hasharon"],
    "kfar saba": ["kfar saba", "kfar sava", "kfar-saba"],
    "raanana": ["raanana", "ra'anana", "ranana"],
    "hod hasharon": ["hod hasharon", "hod ha'sharon"],
    "netanya": ["netanya", "natanya"],
    "haifa": ["haifa", "hefa", "haifa district"],
    "jerusalem": ["jerusalem", "yerushalayim", "jerusalem district"],
    "beer sheva": ["beer sheva", "be'er sheva", "beersheba", "beer-sheva"],
    "rehovot": ["rehovot", "rehovoth"],
    "ness ziona": ["ness ziona", "nes ziona", "nes-ziona"],
    "yokneam": ["yokneam", "yoqneam"],
    "caesarea": ["caesarea", "qesarya"],
    "modiin": ["modiin", "modi'in", "modiin-maccabim-reut"],
    "ashdod": ["ashdod"],
    "ashkelon": ["ashkelon", "ashqelon"],
    "nazareth": ["nazareth", "natzrat"],
    "tiberias": ["tiberias", "teverya"],
    "eilat": ["eilat", "elat"],
    # --- Major global tech hubs ----------------------------------------
    "new york": [
        "new york", "new york city", "nyc", "manhattan", "brooklyn",
        "greater new york", "new york metropolitan",
    ],
    "san francisco": [
        "san francisco", "san francisco bay area", "sf bay area",
        "bay area", "silicon valley",
    ],
    "palo alto": ["palo alto", "mountain view", "menlo park", "sunnyvale"],
    "los angeles": ["los angeles", "greater los angeles", "la metropolitan"],
    "seattle": ["seattle", "greater seattle", "bellevue", "redmond"],
    "boston": ["boston", "greater boston", "cambridge, ma", "cambridge massachusetts"],
    "austin": ["austin", "austin, texas", "austin metropolitan"],
    "london": ["london", "greater london", "london area"],
    "berlin": ["berlin", "berlin metropolitan", "greater berlin"],
    "munich": ["munich", "münchen", "munchen"],
    "amsterdam": ["amsterdam", "greater amsterdam", "amsterdam area"],
    "paris": ["paris", "greater paris", "île-de-france", "ile-de-france"],
    "dublin": ["dublin", "greater dublin"],
    "bengaluru": ["bengaluru", "bangalore"],
    "toronto": ["toronto", "greater toronto", "gta"],
    "singapore": ["singapore"],
}


# ---------------------------------------------------------------------------
# Country → terms (the country's own names + its major cities).
#
# Keyed by the EXACT label used in the Country dropdown. Israel is exhaustive;
# the hub countries get their major cities; every other country at least
# matches its own name.
# ---------------------------------------------------------------------------
def _israel_terms() -> list[str]:
    # NOTE: keep terms specific enough that "%term%" doesn't match foreign
    # places. Deliberately excluded as too-short/ambiguous: "lod" (matches
    # "Lodz, Poland"), "arad" (matches "Arad, Romania"), "acre" (matches
    # "Acre, Brazil" and even mid-word in "Massacre"). Akko/Lod/Arad are
    # small; if needed, a recruiter can type the city explicitly.
    terms = ["israel", "central district", "tel aviv district",
             "southern district", "northern district", "haifa district",
             "jerusalem district", "modiin", "sderot",
             "or yehuda", "rosh haayin", "rosh ha'ayin",
             "kiryat ono", "kiryat gat", "kiryat shmona", "afula",
             "karmiel", "nahariya", "akko", "ramla",
             "dimona", "tirat carmel", "nesher", "migdal haemek",
             "zichron yaakov", "pardes hanna", "hadera", "even yehuda",
             "tel mond", "shoham"]
    # Fold in every curated Israeli city's variations too.
    for city in (
        "tel aviv", "ramat gan", "givatayim", "herzliya", "petah tikva",
        "rishon lezion", "holon", "bat yam", "bnei brak", "ramat hasharon",
        "kfar saba", "raanana", "hod hasharon", "netanya", "haifa",
        "jerusalem", "beer sheva", "rehovot", "ness ziona", "yokneam",
        "caesarea", "modiin", "ashdod", "ashkelon", "nazareth", "tiberias",
        "eilat",
    ):
        terms.extend(CITY_TERMS[city])
    return terms


COUNTRY_TERMS: dict[str, list[str]] = {
    "Israel": _israel_terms(),
    "United States": [
        "united states", "usa", "u.s.", "u.s.a", ", us", "america",
        "new york", "nyc", "san francisco", "bay area", "silicon valley",
        "los angeles", "seattle", "boston", "austin", "chicago", "denver",
        "atlanta", "washington", "dallas", "houston", "miami", "san diego",
        "san jose", "palo alto", "mountain view", "portland", "philadelphia",
    ],
    "United Kingdom": [
        "united kingdom", "uk", "u.k.", "england", "scotland", "wales",
        "london", "manchester", "cambridge", "oxford", "edinburgh",
        "bristol", "leeds", "birmingham", "glasgow",
    ],
    "Germany": [
        "germany", "deutschland", "berlin", "munich", "münchen", "munchen",
        "hamburg", "frankfurt", "cologne", "köln", "koln", "stuttgart",
    ],
    "France": [
        "france", "paris", "île-de-france", "ile-de-france", "lyon",
        "toulouse", "marseille", "bordeaux", "lille",
    ],
    "Netherlands": [
        "netherlands", "holland", "amsterdam", "rotterdam", "the hague",
        "utrecht", "eindhoven",
    ],
    "Ireland": ["ireland", "dublin", "cork", "galway"],
    "Canada": [
        "canada", "toronto", "vancouver", "montreal", "ottawa", "waterloo",
        "calgary",
    ],
    "India": [
        "india", "bengaluru", "bangalore", "hyderabad", "mumbai", "pune",
        "delhi", "gurgaon", "gurugram", "noida", "chennai",
    ],
    "Spain": ["spain", "madrid", "barcelona", "valencia", "málaga", "malaga"],
    "Poland": ["poland", "warsaw", "kraków", "krakow", "wrocław", "wroclaw", "gdańsk", "gdansk"],
    "Switzerland": ["switzerland", "zurich", "zürich", "geneva", "lausanne", "basel"],
    "Sweden": ["sweden", "stockholm", "gothenburg", "malmö", "malmo"],
    "Australia": ["australia", "sydney", "melbourne", "brisbane", "perth"],
    "Singapore": ["singapore"],
    "Ukraine": ["ukraine", "kyiv", "kiev", "lviv", "kharkiv", "dnipro", "odesa", "odessa"],
    "Portugal": ["portugal", "lisbon", "lisboa", "porto"],
    "Brazil": ["brazil", "são paulo", "sao paulo", "rio de janeiro", "belo horizonte"],
}


def _normalize(text: str) -> str:
    # Lowercase + trim, and drop SQL LIKE wildcards (% _ \). Location names
    # never contain them; left in, a typed "%" would turn the DB match into a
    # match-anything wildcard. Stripping keeps the "contains" filter literal.
    cleaned = (text or "").strip().lower()
    for ch in ("%", "_", "\\"):
        cleaned = cleaned.replace(ch, "")
    return cleaned.strip()


def _clean_terms(terms: list[str]) -> list[str]:
    """Normalize, drop empties, and de-dupe while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        t = _normalize(term)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def expand_country(country: str | None) -> list[str]:
    """Return all match terms for a dropdown country, or [] if none/unknown.

    Unknown country (not in the curated map) falls back to matching the
    country's own name.
    """
    if not country:
        return []
    label = country.strip()
    if label in COUNTRY_TERMS:
        return _clean_terms(COUNTRY_TERMS[label])
    norm = _normalize(label)
    return [norm] if norm else []


def expand_city(city: str | None) -> list[str]:
    """Return all match terms for a typed city.

    Looks the city up in the curated variations map (so "Tel Aviv" also
    catches "Tel Aviv-Yafo", "Jaffa", ...). If the city isn't curated, falls
    back to the literal text the user typed — still a valid substring match,
    just without synonyms.
    """
    norm = _normalize(city)
    if not norm:
        return []
    if norm in CITY_TERMS:
        return _clean_terms(CITY_TERMS[norm])
    # Try a light fuzzy hit: typed "tel-aviv" or "telaviv" → "tel aviv".
    collapsed = norm.replace("-", " ").replace(".", " ")
    collapsed = " ".join(collapsed.split())
    if collapsed in CITY_TERMS:
        return _clean_terms(CITY_TERMS[collapsed])
    return [norm]


def expand_location_terms(country: str | None = None, city: str | None = None) -> list[str]:
    """Combine country + city into one deduped, lowercased term list.

    NOTE: this OR-flattens both groups into one list — a profile matches if it
    contains ANY term. Callers that need country AND city to *both* hold (the
    "narrow within country" behaviour) must use ``expand_country`` and
    ``expand_city`` separately and intersect. ``search_similar`` does this.
    Kept for the simple single-group case and for tests.

    Order is preserved (country terms first, then city), duplicates removed.
    """
    return _clean_terms(expand_country(country) + expand_city(city))
