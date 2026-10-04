#! /usr/bin/env bash

# Download all datasets used for classiqa development

DATASETS_DIR=~/Projects/Datasets
mkdir -p $DATASETS_DIR

# KonIQ-10k

echo "Downloading KonIQ-10k..."
mkdir -p ${DATASETS_DIR}/koniq-10k
wget -O ${DATASETS_DIR}/koniq-10k/images.zip "https://datasets.vqa.mmsp-kn.de/archives/koniq10k_512x384.zip"
unzip ${DATASETS_DIR}/koniq-10k/images.zip -d ${DATASETS_DIR}/koniq-10k/images/
rm ${DATASETS_DIR}/koniq-10k/images.zip

wget -O ${DATASETS_DIR}/koniq-10k/scores.zip "https://datasets.vqa.mmsp-kn.de/archives/koniq10k_scores_and_distributions.zip"
unzip ${DATASETS_DIR}/koniq-10k/scores.zip -d ${DATASETS_DIR}/koniq-10k/data
rm ${DATASETS_DIR}/koniq-10k/scores.zip

# Kadid-10k

echo "Downloading Kadid-10k..."
wget -O ${DATASETS_DIR}/kadid10k.zip "https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip"
unzip ${DATASETS_DIR}/kadid10k.zip -d ${DATASETS_DIR}
rm ${DATASETS_DIR}/kadid10k.zip

# CSIQ

echo "Downloading CSIQ..."
mkdir -p ${DATASETS_DIR}/csiq/images/
mkdir -p ${DATASETS_DIR}/csiq/data/
wget -O ${DATASETS_DIR}/csiq/data/csiq.DMOS.xlsx "https://s2.smu.edu/~eclarson/csiq/csiq.DMOS.xlsx"
wget -O ${DATASETS_DIR}/csiq/src_imgs.zip "https://s2.smu.edu/~eclarson/csiq/src_imgs.zip"
wget -O ${DATASETS_DIR}/csiq/dst_imgs.zip "https://s2.smu.edu/~eclarson/csiq/dst_imgs.zip"
unzip ${DATASETS_DIR}/csiq/src_imgs.zip -d ${DATASETS_DIR}/csiq/images/src_imgs
unzip ${DATASETS_DIR}/csiq/dst_imgs.zip -d ${DATASETS_DIR}/csiq/images/dst_imgs
rm ${DATASETS_DIR}/csiq/src_imgs.zip
rm ${DATASETS_DIR}/csiq/dst_imgs.zip

# CID2013

echo "Downloading CID2013..."
wget -O ${DATASETS_DIR}/cid2013.7z "https://zenodo.org/records/2647033/files/CID2013.7z?download=1"
7z x ${DATASETS_DIR}/cid2013.7z -o${DATASETS_DIR}/cid2013
rm ${DATASETS_DIR}/cid2013.7z

# TID2013

echo "Downloading TID2013..."
wget -O ${DATASETS_DIR}/tid2013.rar "http://www.ponomarenko.info/tid2013/tid2013.rar"
unrar x ${DATASETS_DIR}/tid2013.rar ${DATASETS_DIR}/tid2013/
rm ${DATASETS_DIR}/tid2013.rar

# CID:IQ

echo "Downloading CID:IQ..."
wget -O ${DATASETS_DIR}/cidiq.7z "https://zenodo.org/records/17376246/files/CIDIQ.zip?download=1"
7z x ${DATASETS_DIR}/cidiq.7z -o${DATASETS_DIR}/cidiq
rm ${DATASETS_DIR}/cidiq.7z

## NITS-IQA

# For this one you'll need to download the files manully.
# https://drive.google.com/drive/folders/0B_bnn8Xh3PMmT1VxSlVRWDNCTk0?resourcekey=0-9JzjQxVUNJXIodLwkiZ-Lg&usp=drive_link

## LIVE-IQA

# Same here. Go to:
# https://utexas.app.box.com/v/databaserelease2 (images)
# But you'll need to sign the license agreement first. See:
# http://live.ece.utexas.edu/research/quality/subjective.htm

mkdir -p ${DATASETS_DIR}/live-iqa/
wget -O ${DATASETS_DIR}/live-iqa/dmos_realigned.mat "http://live.ece.utexas.edu/research/quality/release2/dmos_realigned.mat"


