#!/bin/sh

echo "setup hadoop warehouse"
spark-submit /home/thanathip/platform/setup-hadoop/create-table.py
echo "finish hadoop warehouse"
 