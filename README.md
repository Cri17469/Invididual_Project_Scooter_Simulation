[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=20899758)
# COMP0035 coursework repository
## Overview
This project analyses Singapore’s *International Visitor Arrivals* datasets (Air and Sea) and builds a normalized SQLite database for long-term tourism insights.  
All scripts are developed in Python following **PEP 8** and **PEP 257** style conventions, and can be executed locally or in GitHub Codespaces.

---

# COMP0035 coursework 1
## Section 1 — Data Description and Preparation

### 1.1 Data Description and Exploration
The script [`Section1_1_describe_data.py`](src/coursework1/Section1/Section1_1_describe_data.py) performs an initial exploration of the two raw datasets located in `data/original_dataset/`.  
Main tasks:
- Display dataset shape, column names, and sample rows.  
- Identify missing values, duplicated rows, and invalid entries (e.g., “na”, “N/A”).  
- Extract the hierarchical structure from the first column (*Total → Region → Country*).  
- Validate that each region’s total equals the sum of its countries.  
- Produce summary statistics and visualisations (e.g., missing-value plots, 2025 distribution charts, country boxplots).  

All exploration steps help verify that the datasets are consistent and ready for transformation.

### 1.2 Data Preparation
The script [`Section1_2_prepared_data.py`](src/coursework1/Section1/Section1_2_prepared_data.py) cleans and restructures both datasets to create a consistent analytical base.  
Key operations:
- Normalise month columns (supporting formats like `2025May`, `2025-05`).  
- Align the region → country mapping between Air and Sea datasets.  
- Validate all hierarchical sums again to ensure accuracy.  
- Export cleaned, ordered CSVs to `data/prepared_dataset/` for later use.  

The prepared data is used as input for **Section 2.2 (Database Creation)**.

---

## Section 2.2 — Database Creation and Verification

### Overview
This stage converts the cleaned **Air** and **Sea** visitor-arrival datasets into a fully-structured **SQLite** database.  
All operations are idempotent, meaning the database can be safely recreated and reloaded multiple times without duplication.

### ⚙️ Scripts and Functions
| Script | Purpose |
|--------|----------|
| [`init_db.py`](src/coursework1/Section2/init_db.py) | Defines the normalized database schema and creates all tables (dimension + fact). |
| [`etl_utils.py`](src/coursework1/Section2/etl_utils.py) | Provides utility functions for ETL: converting wide → long format, inserting dimensions/facts, ensuring month-year consistency. |
| [`load_csv.py`](src/coursework1/Section2/load_csv.py) | Main orchestration script: creates database, resets tables, loads all data from prepared CSVs, and inserts them into SQLite. |
| [`verify_db.py`](src/coursework1/Section2/verify_db.py) | Post-load verification of data integrity, hierarchy consistency, and FK relations. |

### Entity–Relationship Diagram
The ERD is stored in [`Section2/ERD_graph.md`](src/coursework1/Section2/ERD_graph.md).
It defines the normalized structure linking all dimension tables (total, region, country, transport_mode, month_year)
to the main fact table (`monthly_arrivals`).

### Workflow Summary
1. **Read prepared datasets** from `data/prepared_dataset/`.  
2. **Create schema** in `data/database/visitor_arrivals.db` using `init_db.create_database()`.  
3. **Reset tables** safely (FK-aware DELETE order).  
4. **Insert dimensions:** `total`, `region`, `country`, `transport_mode`, and `month_year`.  
5. **Insert fact table:** `monthly_arrivals` for both Air and Sea.  
6. **Verify** database integrity using `verify_db.py`.

### Verification
After the database is created, run:
```bash
python src/coursework1/Section2/verify_db.py
```

## Section 3 — Tools

### 3.1 Environment management  
A Python virtual environment (`.venv`) was used to keep project dependencies consistent.  
Main packages include *pandas*, *matplotlib*, and *sqlite3*, which are listed in `requirements.txt`.  
The same setup can be recreated by following the steps described in the README instructions.  

### 3.2 Source code control  
All project files are version-controlled using Git and hosted on GitHub Classroom.  
Private repository link:  
[https://github.com/KytikE/comp0035-cw-KytikE](https://github.com/KytikE/comp0035-cw-KytikE)

### 3.3 Linting  
Code quality and style were checked with **Pylint** to ensure compliance with PEP 8 and PEP 257.  
The linter reported only minor *duplicate-code* warnings caused by similar `try-except` error-handling blocks across multiple scripts; these are acceptable for consistent exception reporting.  
The final Pylint score was **9.64 / 10**, confirming that the code meets standard Python style guidelines.  
A screenshot of the linter output is included in the report.

---

## Section 4 — References

- **AI use:** ChatGPT (OpenAI ChatGPT) was used only to assist with code documentation wording and report language clarity.  
  All final code, design decisions, and analysis are my own.  
- **Dataset attribution:**  
  The data files are part of the *International Visitor Arrivals* dataset provided through the UCL COMP0035 coursework resources, based on Singapore Tourism Board statistics.  
  Original dataset available from [data.gov.sg](https://data.gov.sg/datasets?page=1&query=tourism&formats=CSV).
- **Other sources:** No additional external code sources were used beyond standard library and package documentation.

---

---
# COMP0035 coursework 2

## Overview
This coursework focuses on applying **object-oriented programming (OOP)** and **automated testing** techniques to a database-backed Python application developed in Coursework 1.

A Python ORM class is implemented using **SQLModel** to represent records from the SQLite database, and a comprehensive automated test suite is provided using the **pytest** framework to verify correctness, validation logic, and database interaction.

---

## Section 2.1 — Python Class

### Overview
In this section, a Python class is implemented to map directly to one of the tables created in the Coursework 1 SQLite database.

The `monthly_arrivals` table was selected because it is the main fact table in the database and contains both numerical values (visitor counts) and foreign key references.  
Using an ORM approach allows database records to be represented and manipulated as Python objects rather than through raw SQL queries.

The implementation uses **SQLModel**, which integrates SQLAlchemy and Pydantic, providing a clean and testable ORM model.

### ORM Class
| File | Purpose |
|------|---------|
| [`section2_1_models.py`](src/coursework2/section2_1_models.py) | Defines the ORM class `MonthlyArrival`, mapping to the `monthly_arrivals` table in the SQLite database. |

### Class Description
The `MonthlyArrival` class represents a single record from the `monthly_arrivals` table.  
Its attributes correspond directly to the table columns, including foreign key identifiers and the visitor count.

Two helper methods are provided to demonstrate behaviour beyond simple data storage:

- `is_valid()`  
  Performs basic application-level validation, such as checking that visitor counts are non-negative and that foreign key identifiers are valid.

- `to_dict()`  
  Converts an ORM object into a plain Python dictionary, which is useful for testing, debugging, and exporting data.

These methods provide clear, testable behaviour and are the primary focus of the automated tests in Section 2.2.

---

## Section 2.2 — Testing

### Overview
This section evaluates the Python class implemented in Section 2.1 using automated tests written with the **pytest** framework.

The purpose of the testing is to demonstrate that:
- The ORM class is correctly mapped to the underlying database table.
- The helper methods behave as expected.
- The testing setup can be executed reliably by the marker.

### Test Scope
A total of **12 tests** were written to test the `MonthlyArrival` ORM class defined in Section 2.1.

All tests focus **only** on this class, as required by the coursework brief.  
No controller, dashboard, or presentation-layer logic is tested.

### Test Coverage
The tests cover three main aspects:

#### ORM mapping and database access
Several tests verify that:
- The ORM model is correctly mapped to the `monthly_arrivals` table.
- Records can be queried successfully using SQLModel sessions.

These tests ensure correct integration with the existing SQLite database created in Coursework 1.

#### Helper method behaviour (unit tests)
The helper methods are tested independently:
- `to_dict()` returns a dictionary with the expected keys and values.
- `is_valid()` correctly accepts valid objects and rejects invalid ones.

These tests are **pure unit tests**, as they operate on Python objects without relying on database state.

#### Validation and error handling
Additional tests deliberately construct invalid ORM objects to ensure that validation logic behaves correctly.  
Examples include negative foreign key values or invalid visitor counts.

Each test function includes a docstring explicitly stating whether it is a unit test or an integration-style test involving the database, in line with the coursework requirements.

---

### Running the Tests

The marker can run the tests using the standard commands specified in the coursework brief:

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest
