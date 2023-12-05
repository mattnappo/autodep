#!/bin/bash

ali --duration=5s \
    --rate=35 \
    --workers=12 \
    --header 'Content-Type: application/json' \
    --body-file=$1 \
    --method=POST http://localhost:9000/inference

