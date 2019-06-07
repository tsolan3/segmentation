#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

./manage.py migrate
./manage.py set_social_account_credentials
./manage.py collectstatic --noinput
gunicorn itrainyou.wsgi:application -c /home/app/config/gunicorn.conf.py
