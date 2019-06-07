from multiprocessing import cpu_count

bind = '0.0.0.0:8000'
max_requests = 1000
worker_class = 'sync'
workers = cpu_count() + 1
reload = True
name = 'SpotPet'
errorlog = '/var/log/gunicorn-error.log'
loglevel = 'critical'
