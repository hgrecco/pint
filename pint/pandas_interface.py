import sys

class PintPandasImporter():
    
    def __init__(self):
        try:
            import pintpandas
            self.HAS_PINTPANDAS = True
            self.import_error=""
        except Exception:
            _, err, _ = sys.exc_info()
            self.HAS_PINTPANDAS = False
            self.import_error = str(err)
        
    def __repr__(self):
        return str(self.HAS_PINTPANDAS)+self.import_error
        
PandasInterface = PintPandasImporter()
