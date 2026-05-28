class SimpleMemory:

    def __init__(self):

        self.chat_history = []

    # ==========================================
    # SAVE CONTEXT
    # ==========================================

    def save_context(

        self,

        inputs,

        outputs
    ):

        self.chat_history.append({

            "input": inputs,

            "output": outputs
        })

    # ==========================================
    # LOAD MEMORY
    # ==========================================

    def load_memory_variables(
        self
    ):

        return {

            "history": self.chat_history
        }

    # ==========================================
    # CLEAR MEMORY
    # ==========================================

    def clear(self):

        self.chat_history = []

# ==============================================
# MEMORY FACTORY
# ==============================================

def get_memory():

    return SimpleMemory()