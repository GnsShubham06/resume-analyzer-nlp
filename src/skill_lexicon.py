"""Hand-built skill / qualification / job-title lexicons.

These lists drive (1) dictionary-based skill extraction, (2) skill-aware
multi-word tokenization and (3) the rule-based *content check* used to flag
resumes whose content does not match their folder label.  They are NOT used
as features by the classifiers, so classification and extraction stay
separate components.
"""

DOMAIN_SKILLS = {
    "Engineering": [
        "autocad", "solidworks", "catia", "ptc creo", "autodesk inventor", "ansys", "abaqus",
        "hypermesh", "finite element analysis", "fea", "cad", "cam", "cae", "gd&t",
        "mechanical design", "product design", "process engineering", "industrial automation",
        "plc", "plc programming", "scada", "hmi", "controls engineering", "electrical design",
        "root cause analysis", "lean manufacturing", "six sigma", "kaizen", "5s", "dfmea", "pfmea",
        "design for manufacturing", "bill of materials", "prototyping", "3d printing", "welding",
        "machining", "calibration", "test instruments", "quality control", "quality assurance",
        "preventive maintenance", "troubleshooting", "hydraulics", "pneumatics", "matlab",
        "labview", "python", "c++", "java", "c#", "sql", "linux", "software development",
        "software engineering", "agile", "scrum", "devops", "embedded systems", "aerospace",
        "manufacturing", "project engineering", "systems engineering", "sheet metal",
        "tolerance analysis", "design optimization", "machine learning", "data analysis",
        "network administration", "technical support", "sdlc", "software development life cycle",
    ],
    "HR": [
        "human resources", "human resource management", "recruitment", "recruiting",
        "talent acquisition", "onboarding", "orientation", "employee relations",
        "performance management", "performance appraisal", "compensation", "benefits administration",
        "benefits", "payroll", "adp", "hris", "peoplesoft", "workday", "kronos", "fmla",
        "leave administration", "employee training", "training and development", "succession planning",
        "workforce planning", "labor relations", "conflict management", "conflict resolution",
        "employment law", "eeo", "affirmative action", "background checks", "i-9", "job descriptions",
        "interviewing", "personnel", "personnel management", "hr policies", "employee engagement",
        "exit interviews", "retention", "hr generalist", "staffing", "time keeping", "timekeeping",
        "data entry", "file management", "office management", "administrative support", "visa processing",
    ],
    "Finance": [
        "financial analysis", "financial reporting", "financial statements", "financial management",
        "accounting", "general ledger", "accounts payable", "accounts receivable", "budgeting",
        "forecasting", "budgeting and forecasting", "budget preparation", "cash flow", "cash management",
        "reconciliation", "account reconciliation", "audit", "auditing", "internal audit", "tax",
        "tax preparation", "gaap", "ifrs", "sox", "variance analysis", "cost accounting",
        "fund accounting", "investment", "investment reporting", "financial modeling", "valuation",
        "risk management", "treasury", "quickbooks", "sap", "oracle financials",
        "month-end close", "year-end close", "payroll processing", "cpa", "cfa", "bookkeeping",
        "financial planning", "fp&a", "credit analysis", "banking", "invoicing", "billing",
        "profit and loss", "p&l", "balance sheet", "grant accounting", "cost control",
    ],
    "Healthcare": [
        "patient care", "patient-centered care", "nursing", "registered nurse", "rn", "lpn", "cna",
        "bls", "acls", "cpr", "emr", "ehr", "epic", "cerner", "hipaa", "medical terminology",
        "clinical", "clinical research", "clinical trials", "clinical supervision", "psychotherapy",
        "behavioral health", "mental health", "case management", "medication administration",
        "vital signs", "phlebotomy", "triage", "oncology", "pharmacy", "pharmaceutical",
        "pharmaceutical sales", "medical devices", "healthcare", "health care", "hospital",
        "patient access", "patient registration", "medical billing", "medical coding", "icd-10",
        "infection control", "evidence based practice", "care planning", "discharge planning",
        "regulatory affairs", "fda", "oral care", "social work", "lcsw", "crisis intervention",
    ],
    "Agriculture": [
        "agriculture", "agronomy", "crop production", "crop management", "soil science", "soil testing",
        "irrigation", "horticulture", "plant protection", "plant pathology", "entomology",
        "pest management", "integrated pest management", "ipm", "pesticide application", "quarantine",
        "invasive species", "farm management", "livestock", "animal husbandry", "veterinary",
        "greenhouse", "nursery", "forestry", "wetlands", "wetland delineation", "environmental biology",
        "ecology", "field research", "field data collection", "biodiversity", "species identification",
        "usda", "aphis", "agricultural extension", "seed", "fertilizer", "harvest", "tractor",
        "food safety", "grain", "dairy", "poultry", "lepidoptera", "botany", "gis",
    ],
}

GENERAL_SKILLS = [
    "microsoft office", "ms office", "microsoft word", "word", "powerpoint", "outlook",
    "ms excel", "microsoft excel", "customer service", "communication skills", "leadership",
    "team leadership", "project management", "time management", "problem solving",
    "critical thinking", "public speaking", "event planning", "documentation", "scheduling",
    "budget management", "strategic planning", "business development", "sales", "marketing",
    "inventory management", "vendor management", "contract negotiation", "data analysis",
    "report writing", "presentation skills", "training", "supervision", "process improvement",
]

DEGREE_PATTERNS = [
    r"\b(ph\.?\s?d\.?|doctor(?:ate)? of [a-z ]+)",
    r"\b(m\.?b\.?a\.?|master of business administration)\b",
    r"\b(m\.?s\.?c?\.?|m\.?a\.?|m\.?eng\.?|m\.?tech|master(?:'s)?(?: of| in)? [a-z ]+)\b",
    r"\b(b\.?s\.?c?\.?|b\.?a\.?|b\.?e\.?|b\.?tech|b\.?eng\.?|b\.?b\.?a\.?|bachelor(?:'s)?(?: of| in)? [a-z ]+)\b",
    r"\b(associate(?:'s)? (?:of|in|degree)[a-z ]*|a\.?a\.?s\.?)\b",
    r"\b(high school diploma|ged|diploma(?: in [a-z ]+)?)\b",
    r"\b(certificat(?:e|ion) (?:in|of) [a-z ]+)\b",
]

QUALIFICATION_KEYWORDS = [
    "cpa", "cfa", "pmp", "shrm-cp", "shrm-scp", "phr", "sphr", "rn", "lpn", "lcsw", "cna",
    "six sigma green belt", "six sigma black belt", "eit", "cma", "bls", "acls",
]

JOB_TITLE_HEADWORDS = [
    "engineer", "technician", "manager", "director", "analyst", "specialist", "coordinator",
    "assistant", "generalist", "accountant", "controller", "officer", "nurse", "representative",
    "administrator", "supervisor", "consultant", "intern", "clerk", "executive", "president",
    "principal", "lead", "developer", "scientist", "biologist", "therapist", "counselor",
    "recruiter", "auditor", "bookkeeper", "associate", "agent", "mechanic", "operator",
    "rep", "head", "chief", "vp", "partner", "teacher", "instructor", "programmer", "architect",
]

SECTION_HEADERS = [
    "summary", "professional summary", "executive profile", "professional profile", "profile",
    "career overview", "career focus", "objective", "professional overview", "highlights",
    "skill highlights", "skills", "core qualifications", "qualifications", "technical skills",
    "experience", "professional experience", "work experience", "work history", "employment history",
    "education", "education and training", "certifications", "licenses", "license",
    "accomplishments", "achievements", "additional information", "interests", "languages",
    "affiliations", "professional affiliations", "presentations", "publications", "training",
    "volunteer experience", "military experience", "awards", "activities", "references",
]


def all_skill_terms() -> dict[str, str]:
    """Map every skill term -> its domain ("General" for general skills)."""
    terms = {}
    for t in GENERAL_SKILLS:
        terms[t] = "General"
    for dom, lst in DOMAIN_SKILLS.items():
        for t in lst:
            terms[t] = dom
    return terms
