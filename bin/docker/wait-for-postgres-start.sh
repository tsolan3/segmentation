#!/bin/bash

set -o errexit
set -o pipefail

cmd="$@"

function postgres_ready(){
python << END
import sys
import psycopg2
try:
    conn = psycopg2.connect(dbname="$POSTGRES_DB", user="$POSTGRES_USER", password="$POSTGRES_PASSWORD", host="postgres")
except psycopg2.OperationalError:
    sys.exit(-1)
sys.exit(0)
END
}

until postgres_ready; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 10
done

>&2 echo "Postgres is up - continuing..."
exec $cmd