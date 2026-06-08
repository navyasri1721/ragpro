from db.mysql_connection import get_connection

def search_mysql(question):

    try:

        conn = get_connection()

        cursor = conn.cursor(dictionary=True)

        query = question.lower()

        remove_words = [
            "who",
            "what",
            "is",
            "the",
            "of",
            "ceo",
            "package",
            "cgpa",
            "required"
        ]

        for word in remove_words:

            query = query.replace(word, "")

        query = query.strip()

        sql = """
        SELECT *
        FROM companies
        WHERE LOWER(company_name) LIKE %s
        LIMIT 5
        """

        keyword = f"%{query}%"

        cursor.execute(
            sql,
            (keyword,)
        )

        results = cursor.fetchall()

        cursor.close()

        conn.close()

        return results

    except Exception as e:

        print("MYSQL ERROR:", e)

        return []