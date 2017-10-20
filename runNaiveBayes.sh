#!/bin/bash

CUR_DIR=`pwd`
SRC_DIR=${CUR_DIR}/src/
OUT_DIR=${CUR_DIR}/out/

javac -d ${OUT_DIR} ${SRC_DIR}/*.java

java -cp ${OUT_DIR} NaiveBayes