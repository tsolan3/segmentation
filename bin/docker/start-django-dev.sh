#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

./manage.py migrate
./manage.py set_social_account_credentials
./manage.py runserver 0.0.0.0:8000