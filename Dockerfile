FROM python:3.6
ENV PYTHONUNBUFFERED 1

RUN apt-get -qq update && apt-get -y install binutils libproj-dev gdal-bin gettext

RUN mkdir /home/app \
 && mkdir /home/app/config \
 && mkdir -p /var/spot_pet/ \
 && mkdir -p /var/spot_pet/static/ \
 && mkdir -p /var/spot_pet/media/

ADD /config/requirements.txt /home/app/config/
RUN pip install -r /home/app/config/requirements.txt

COPY . /home/app/

WORKDIR /home/app/src

ENTRYPOINT ["/home/app/bin/docker/wait-for-postgres-start.sh"]