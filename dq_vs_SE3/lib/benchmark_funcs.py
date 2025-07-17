import tracemalloc

# --- Memory Usage Tracking ---
def measure_peak_memory(func):
    '''
    Return:
        KB of RAM consumed
    '''
    tracemalloc.start()
    func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024  # KB