#!/bin/bash

grep 'PAE,.*PAE' $1 | sed -E 's/.*PAE,//; s/,PAE.*//' > $2.csv