#!/bin/bash

rm -f '/tmp/itrainyou.pid'
celery -A itrainyou.celery_app beat --pidfile=/tmp/itrainyou.pid -s /var/celerybeat-schedule
