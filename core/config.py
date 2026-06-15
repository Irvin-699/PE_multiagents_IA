TICKERS: dict[str, str] = {
    "Nasdaq-100":           "^NDX",      # Invesco    — depuis 1985 · Actions US tech
    "MSCI EM IMI":          "EIMI.L",    # iShares    — depuis 2014 · Actions émergentes
    "STOXX Eur600 Finance": "EXH2.DE",   # iShares    — depuis 2001 · Actions Europe finance
    "Global Healthcare":    "IXJ",       # iShares    — depuis 2001 · Actions santé mondiale
    "US Treasury 10+Y":     "LUTR.L",    # SPDR       — depuis 2016 · Obligations LT US
    "Bund 1-3Y":            "DBXP.DE",   # Xtrackers  — depuis 2007 · Obligations CT Europe
    "Clean Energy":         "ICLN",      # iShares    — depuis 2008 · Énergie propre ESG
    "Global REITs":         "EPRA.PA",   # Amundi     — depuis 2016 · Immobilier monde
    "Gold":                 "GLD",       # SPDR       — depuis 2004 · Or physique
    "Bitcoin":              "BTC-USD",   # Yahoo/CG   — depuis 2014 · Satellite crypto
}

# Période commune à tous les actifs (contrainte LUTR.L + EPRA.PA, disponibles depuis 2016)
START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"
