#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

celery worker -A itrainyou.celery_app