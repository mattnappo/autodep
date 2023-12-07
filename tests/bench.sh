#!/bin/bash

ali --duration=5s \
    --rate=25 \
    --workers=12 \
    --header 'Content-Type: application/json' \
    --body-file=tests/classification_req.json \
    --method=POST http://localhost:9000/inference

