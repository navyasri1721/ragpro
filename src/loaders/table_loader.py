import pdfplumber
import pandas as pd

from io import BytesIO

from langchain_core.documents import (
    Document
)


# =========================================================
# SAFE METADATA CLEANER
# =========================================================

def clean_metadata_value(value):

    try:

        # =============================================
        # NONE
        # =============================================

        if value is None:
            return ""

        # =============================================
        # NaN
        # =============================================

        if pd.isna(value):
            return ""

        # =============================================
        # BASIC TYPES
        # =============================================

        if isinstance(

            value,

            (
                str,
                int,
                float,
                bool
            )
        ):

            return value

        # =============================================
        # EVERYTHING ELSE
        # =============================================

        return str(value)

    except:

        return str(value)


# =========================================================
# TABLE EXTRACTION
# =========================================================

def extract_tables_from_pdf(

    file_bytes,

    filename="file.pdf"
):

    docs = []

    try:

        with pdfplumber.open(
            BytesIO(file_bytes)
        ) as pdf:

            for page_num, page in enumerate(
                pdf.pages
            ):

                tables = page.extract_tables()

                if not tables:
                    continue

                # =====================================
                # PROCESS TABLES
                # =====================================

                for table_idx, table in enumerate(
                    tables
                ):

                    try:

                        if (
                            not table
                            or len(table) < 2
                        ):
                            continue

                        headers = table[0]

                        rows = table[1:]

                        # =================================
                        # CLEAN HEADERS
                        # =================================

                        cleaned_headers = []

                        for h in headers:

                            if h is None:

                                cleaned_headers.append(
                                    "column"
                                )

                            else:

                                cleaned_headers.append(
                                    str(h).strip()
                                )

                        # =================================
                        # DATAFRAME
                        # =================================

                        df = pd.DataFrame(

                            rows,

                            columns=cleaned_headers
                        )

                        # =================================
                        # EACH ROW = ONE DOC
                        # =================================

                        for row_idx, row in df.iterrows():

                            semantic_parts = []

                            safe_metadata = {

                                "source": filename,

                                "type": "table",

                                "page": int(
                                    page_num + 1
                                ),

                                "table_index": int(
                                    table_idx
                                ),

                                "row_index": int(
                                    row_idx
                                )
                            }

                            company_name = None

                            # =============================
                            # BUILD SEMANTIC ROW
                            # =============================

                            for col in df.columns:

                                try:

                                    value = row[col]

                                    value = clean_metadata_value(
                                        value
                                    )

                                    col_name = str(
                                        col
                                    ).strip()

                                    if not value:
                                        continue

                                    semantic_parts.append(
                                        f"{col_name} is {value}"
                                    )

                                    # =====================
                                    # SAFE METADATA
                                    # =====================

                                    safe_metadata[
                                        col_name
                                    ] = value

                                    # =====================
                                    # COMPANY DETECTION
                                    # =====================

                                    if (

                                        col_name.lower()

                                        in [

                                            "company",

                                            "company name",

                                            "name"
                                        ]
                                    ):

                                        company_name = value

                                except Exception as e:

                                    print(
                                        f"ROW FIELD ERROR: {e}"
                                    )

                            # =============================
                            # FINAL TEXT
                            # =============================

                            semantic_text = " ; ".join(
                                semantic_parts
                            )

                            if len(
                                semantic_text.strip()
                            ) < 10:

                                continue

                            safe_metadata[
                                "company"
                            ] = company_name or ""

                            # =============================
                            # CREATE DOCUMENT
                            # =============================

                            docs.append(

                                Document(

                                    page_content=semantic_text,

                                    metadata=safe_metadata
                                )
                            )

                    except Exception as e:

                        print(
                            f"TABLE PROCESS ERROR: {e}"
                        )

    except Exception as e:

        print(
            f"TABLE EXTRACTION ERROR: {e}"
        )

    print(
        f"[DEBUG] TABLE DOCS CREATED: {len(docs)}"
    )

    return docs