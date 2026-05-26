import multiprocessing

workers = max(2, multiprocessing.cpu_count() * 2 + 1)
timeout = 120
accesslog = '-'  # stdout
errorlog = '-'
loglevel = 'info'
