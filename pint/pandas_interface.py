import sys

def load_pintpandas():
    try:
        import pintpandas
        return pintpandas
    except Exception:
        _, err, _ = sys.exc_info()
        return str(err)
    
pintpandas = load_pintpandas()
