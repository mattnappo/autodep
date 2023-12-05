#!/bin/bash

ali --duration=10s \
    --rate=50 \
    --workers=8 \
    --header 'Content-Type: application/json' \
    --body-file=$1 \
    --method=POST http://localhost:9000/inference

