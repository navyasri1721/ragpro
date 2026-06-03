from db.mysql_connection import get_connection


def save_chat(user_message: str, bot_response: str):
    """
    Save a single chat interaction into MySQL database
    """

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO chat_history (user_message, bot_response)
        VALUES (%s, %s)
        """

        cursor.execute(query, (user_message, bot_response))
        conn.commit()

    except Exception as e:
        print(f"[ERROR] Failed to save chat: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()